from typing import List, Sequence

from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCallResult,
)
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemory
from llama_index.core.tools import AsyncBaseTool
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent


class MyFunctionAgent(FunctionAgent):
    async def take_step(
        self,
        ctx: Context,
        llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
        memory: BaseMemory,
    ) -> AgentOutput:
        """Take a single step with the function calling agent."""
        if not self.llm.metadata.is_function_calling_model:
            raise ValueError("LLM must be a FunctionCallingLLM")

        scratchpad: List[ChatMessage] = await ctx.get(self.scratchpad_key, default=[])
        current_llm_input = [*llm_input, *scratchpad]

        ctx.write_event_to_stream(
            AgentInput(input=current_llm_input, current_agent_name=self.name)
        )
        print(f"==========>Available tools: {[tool.metadata.name for tool in tools]}")
        response = await self.llm.astream_chat_with_tools(  # type: ignore
            tools, chat_history=current_llm_input, allow_parallel_tool_calls=True
        )
        async for r in response:
            tool_calls = self.llm.get_tool_calls_from_response(  # type: ignore
                r, error_on_no_tool_call=False
            )
            raw = r.raw.model_dump() if isinstance(r.raw, BaseModel) else r.raw
            ctx.write_event_to_stream(
                AgentStream(
                    delta=r.delta or "",
                    response=r.message.content or "",
                    tool_calls=tool_calls or [],
                    raw=raw,
                    current_agent_name=self.name,
                )
            )

        tool_calls = self.llm.get_tool_calls_from_response(  # type: ignore
            r, error_on_no_tool_call=False
        )
        print(f"======================>Tool calls: {tool_calls}")
        # only add to scratchpad if we didn't select the handoff tool
        scratchpad.append(r.message)
        await ctx.set(self.scratchpad_key, scratchpad)

        raw = r.raw.model_dump() if isinstance(r.raw, BaseModel) else r.raw
        return AgentOutput(
            response=r.message,
            tool_calls=tool_calls or [],
            raw=raw,
            current_agent_name=self.name,
        )
