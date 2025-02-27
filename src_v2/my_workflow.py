from llama_index.core.agent.workflow import AgentWorkflow
from typing import Optional

from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.tools import (
    AsyncBaseTool,
    FunctionTool,
)
from llama_index.core.workflow import (
    Context
)

DEFAULT_HANDOFF_OUTPUT_PROMPT = (
    "Agent {to_agent} is now handling the request due to the following reason: {reason}.\n"
    "Please continue with the current request: \n"
    "> {user_last_request}"
)


async def handoff(ctx: Context, to_agent: str, reason: str, user_last_request: str) -> str:
    """Handoff control of that chat to the given agent."""
    agents: list[str] = await ctx.get("agents")
    current_agent_name: str = await ctx.get("current_agent_name")
    if to_agent not in agents:
        valid_agents = ", ".join([x for x in agents if x != current_agent_name])
        return f"Agent {to_agent} not found. Please select a valid agent to hand off to. Valid agents: {valid_agents}"

    await ctx.set("next_agent", to_agent)
    handoff_output_prompt = await ctx.get(
        "handoff_output_prompt", default=DEFAULT_HANDOFF_OUTPUT_PROMPT
    )

    return handoff_output_prompt.format(to_agent=to_agent, reason=reason, user_last_request=user_last_request)


class MyAgentWorkflow(AgentWorkflow):
    def _get_handoff_tool(
        self, current_agent: BaseWorkflowAgent
    ) -> Optional[AsyncBaseTool]:
        """Creates a handoff tool for the given agent."""
        agent_info = {cfg.name: cfg.description for cfg in self.agents.values()}

        # Filter out agents that the current agent cannot handoff to
        configs_to_remove = []
        for name in agent_info:
            if name == current_agent.name:
                configs_to_remove.append(name)
            elif (
                current_agent.can_handoff_to is not None
                and name not in current_agent.can_handoff_to
            ):
                configs_to_remove.append(name)

        for name in configs_to_remove:
            agent_info.pop(name)

        if not agent_info:
            return None

        fn_tool_prompt = self.handoff_prompt.format(agent_info=str(agent_info))
        return FunctionTool.from_defaults(
            async_fn=handoff, description=fn_tool_prompt, return_direct=False
        )

    # def _get_handoff_tool(
    #     self, current_agent: BaseWorkflowAgent
    # ) -> Optional[AsyncBaseTool]:
    #     handoff_tool: FunctionTool = super()._get_handoff_tool(current_agent)
    #     handoff_tool._metadata.return_direct = False
    #     return handoff_tool
