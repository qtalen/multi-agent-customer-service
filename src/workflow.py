from llama_index.core import Settings
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)
from llama_index.core.tools import (
    BaseTool,
    ToolSelection
)
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.program.function_program import get_function_tool
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai_like import OpenAILike

from agents import (
    RequestTransfer,
    AgentConfig,
    TransferToAgent,
    get_agent_config_pair,
    get_agent_configs_str
)
from utils import FunctionToolWithContext

ORCHESTRATION_PROMPT = """  
    You are a customer service manager for a drone store.
    Based on the user's current status, latest request, and the available customer service agents, you help the user decide which agent to consult next.
    
    You don't focus on the dependencies between agents; the agents will handle those themselves.
    If the user asks about something unrelated to drones, you should politely and briefly decline to answer.
    
    Here is the list of available customer service agents:
    {agent_configs_str}
    
    Here is the user's current status:
    {user_state_str}
"""


class OrchestrationEvent(Event):
    query: str


class ActiveSpeakerEvent(Event):
    query: str


class ToolCallEvent(Event):
    tool_call: ToolSelection
    tools: list[BaseTool]


class ToolCallResultEvent(Event):
    chat_message: ChatMessage


class ProgressEvent(Event):
    msg: str


class CustomerService(Workflow):
    def __init__(
            self,
            llm: OpenAILike | None = None,
            memory: ChatMemoryBuffer = None,
            user_state: dict[str, str | None] = None,
            *args,
            **kwargs
    ):
        self.llm = llm or Settings.llm
        self.memory = memory or ChatMemoryBuffer()
        self.user_state = user_state
        super().__init__(*args, **kwargs)

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="We're making some progress."))
        return StopEvent(result="Hello World")

    @step
    async def start(
            self, ctx: Context, ev: StartEvent
    ) -> ActiveSpeakerEvent | OrchestrationEvent:
        self.memory.put(ChatMessage(
            role="user",
            content=ev.msg
        ))
        user_state = await ctx.get("user_state", None)
        if not user_state:
            await ctx.set("user_state", self.user_state)

        user_msg = ev.msg
        active_speaker = await ctx.get("active_speaker", default=None)

        if active_speaker:
            return ActiveSpeakerEvent(query=user_msg)
        else:
            return OrchestrationEvent(query=user_msg)

    @step
    async def speak_with_sub_agent(
            self, ctx: Context, ev: ActiveSpeakerEvent
    ) -> OrchestrationEvent | ToolCallEvent | StopEvent:
        active_speaker = await ctx.get("active_speaker", default="")
        agent_config: AgentConfig = get_agent_config_pair()[active_speaker]
        chat_history = self.memory.get()
        user_state_str = await self._get_user_state_str(ctx)

        system_prompt = (
                agent_config.system_prompt.strip()
                + f"\n\n<user state>:\n{user_state_str}"
        )
        ctx.write_event_to_stream(ProgressEvent(msg=f"In step speak_with_sub_agent, system_prompt:\n{system_prompt}"))
        llm_input = [ChatMessage(role="system", content=system_prompt)] + chat_history
        tools = [get_function_tool(RequestTransfer)] + agent_config.tools
        event, tool_calls, response = await self.achat_to_tool_calls(ctx, tools, llm_input)

        if event is not None:
            return event
        await ctx.set("num_tool_calls", len(tool_calls))
        for tool_call in tool_calls:
            if tool_call.tool_name == "RequestTransfer":
                await ctx.set("active_speaker", None)
                ctx.write_event_to_stream(
                    ProgressEvent(msg="The agent is requesting a transfer, please hold on...")
                )
                return OrchestrationEvent(query=ev.query)
            else:
                ctx.send_event(
                    ToolCallEvent(tool_call=tool_call, tools=agent_config.tools)
                )
        await self.memory.aput(response.message)

    @step(num_workers=4)
    async def handle_tool_calls(
            self, ctx: Context, ev: ToolCallEvent
    ) -> ToolCallResultEvent:
        tool_call = ev.tool_call
        tools_by_name = {tool.metadata.get_name(): tool for tool in ev.tools}
        tool_msg = None
        tool = tools_by_name[tool_call.tool_name]
        additional_kwargs = {
            "tool_call_id": tool_call.tool_id,
            "name": tool.metadata.get_name()
        }
        if not tool:
            tool_msg = ChatMessage(
                role="tool",
                content=f"Tool {tool_call.tool_name} does not exists.",
                additional_kwargs=additional_kwargs
            )
            return ToolCallResultEvent(chat_message=tool_msg)

        try:
            if isinstance(tool, FunctionToolWithContext):
                tool_output = await tool.acall(ctx, **tool_call.tool_kwargs)
            else:
                tool_output = await tool.acall(**tool_call.tool_kwargs)

            tool_msg = ChatMessage(
                role="tool",
                content=tool_output.content,
                additional_kwargs=additional_kwargs
            )
        except Exception as e:
            tool_msg = ChatMessage(
                role="tool",
                content=f"Encountered error in tool call: {e}",
                additional_kwargs=additional_kwargs
            )
        return ToolCallResultEvent(chat_message=tool_msg)

    @step
    async def aggregate_tool_results(
            self, ctx: Context, ev: ToolCallResultEvent
    ) -> ActiveSpeakerEvent | None:
        num_tool_calls = await ctx.get("num_tool_calls")
        results = ctx.collect_events(ev, [ToolCallResultEvent] * num_tool_calls)
        if not results:
            return None

        for result in results:
            await self.memory.aput(result.chat_message)
        return ActiveSpeakerEvent(query="")

    @step
    async def orchestrate(
            self, ctx: Context, ev: OrchestrationEvent
    ) -> ActiveSpeakerEvent | StopEvent:
        self.memory.reset()
        await self.memory.aput(ChatMessage(
            role="user", content=ev.query
        ))
        chat_history = self.memory.get()
        user_state_str = await self._get_user_state_str(ctx)
        system_prompt = ORCHESTRATION_PROMPT.format(
            agent_configs_str=get_agent_configs_str(),
            user_state_str=user_state_str
        )
        messages = [ChatMessage(role="system", content=system_prompt)] + chat_history
        tools = [get_function_tool(TransferToAgent)]
        event, tool_calls, _ = await self.achat_to_tool_calls(ctx, tools, messages)
        if event is not None:
            return event
        tool_call = tool_calls[0]
        selected_agent = tool_call.tool_kwargs["agent_name"]
        await ctx.set("active_speaker", selected_agent)
        ctx.write_event_to_stream(
            ProgressEvent(msg=f"In step orchestrate:\nTransfer to agent: {selected_agent}")
        )
        return ActiveSpeakerEvent(query=ev.query)

    async def achat_to_tool_calls(self,
                                  ctx: Context,
                                  tools: list[FunctionTool],
                                  chat_history: list[ChatMessage]
        ) -> tuple[StopEvent | None, list[ToolSelection], ChatResponse]:
        response = await self.llm.achat_with_tools(tools, chat_history=chat_history)
        tool_calls: list[ToolSelection] = self.llm.get_tool_calls_from_response(
            response=response, error_on_no_tool_call=False
        )
        stop_event = None
        if len(tool_calls) == 0:
            await self.memory.aput(response.message)
            stop_event = StopEvent(
                result=response.message.content
            )
        return stop_event, tool_calls, response

    @staticmethod
    async def _get_user_state_str(ctx: Context) -> str:
        user_state = await ctx.get("user_state", None)
        user_state_list = [f"{k}: {v}" for k, v in user_state.items()]
        return "\n".join(user_state_list)
