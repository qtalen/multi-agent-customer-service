from pydantic import BaseModel, Field, ConfigDict
from llama_index.core.workflow import Context
from llama_index.core.tools import BaseTool

from utils import FunctionToolWithContext
from models import INDEXES, query_docs

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


class AgentConfig(BaseModel):
    """
    Detailed configuration for an agent
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(description="agent name")
    description: str = Field(
        description="agent description, which describes what the agent does"
    )
    system_prompt: str | None = None
    tools: list[BaseTool] | None = Field(
        description="function tools available for this agent"
    )


class TransferToAgent(BaseModel):
    """Used to explain which agent to transfer to next."""
    agent_name: str = Field(description="The name of the agent to transfer to.")


class RequestTransfer(BaseModel):
    """
    Used to indicate that you don't have the necessary permission to complete the user's request,
    or that you've already completed the user's request and want to transfer to another agent.
    """
    pass


def get_authentication_tools() -> list[BaseTool]:
    async def login(ctx: Context, username: str) -> bool:
        """When the user provides their name, you can use this method to update their status.。
        :param username The user's title or name.
        """
        if not username:
            return False

        user_state = await ctx.get("user_state", None)
        user_state["name"] = username.strip()
        await ctx.set("user_state", user_state)
        return True

    return [FunctionToolWithContext.from_defaults(async_fn=login)]


def get_pre_sales_tools() -> list[BaseTool]:
    async def skus_info_retrieve(ctx: Context, query: str) -> str:
        """
        When the user asks about a product, you can use this tool to look it up.
        :param query: The user's request.
        :return: The information found.
        """
        sku_info = await query_docs(INDEXES["SKUS"], query)
        return sku_info

    return [FunctionToolWithContext.from_defaults(async_fn=skus_info_retrieve)]


def get_after_sales_tools() -> list[BaseTool]:
    async def terms_info_retrieve(ctx: Context, query: str) -> str:
        """
        When the user asks about how to use a product, or about after-sales and repair options, you can use this tool to look it up.
        :param query: The user's request.
        :return: The information found.
        """
        terms_info = await query_docs(INDEXES["TERMS"], query)
        return terms_info
    return [FunctionToolWithContext.from_defaults(async_fn=terms_info_retrieve)]


def _get_agent_configs() -> list[AgentConfig]:
    return [
        AgentConfig(
            name="Authentication Agent",
            description="Record the user's name. If there's no name, you need to ask this from the customer.",
            system_prompt="""
            You are a front desk customer service agent for registration.
            If the user hasn't provided their name, you need to ask them.
            When the user has other requests, transfer the user's request.
            """,
            tools=get_authentication_tools()
        ),
        AgentConfig(
            name="Pre Sales Agent",
            description="When the user asks about product information, you need to consult this customer service agent.",
            system_prompt="""
            You are a customer service agent answering pre-sales questions for customers.
            You will respond to users' inquiries based on the context of the conversation.
            
            When the context is not enough, you will use tools to supplement the information.
            You can only handle user inquiries related to product pre-sales. 
            Please use the RequestTransfer tool to transfer other user requests.
            """,
            tools=get_pre_sales_tools()
        ),
        AgentConfig(
            name="After Sales Agent",
            description="When the user asks about after-sales information, you need to consult this customer service agent.",
            system_prompt="""
            You are a customer service agent answering after-sales questions for customers, including how to use the product, return and exchange policies, and repair solutions.
            You respond to users' inquiries based on the context of the conversation.
            When the context is not enough, you will use tools to supplement the information.
            You can only handle user inquiries related to product after-sales. 
            Please use the RequestTransfer tool to transfer other user requests.
            """,
            tools=get_after_sales_tools()
        )
    ]


def get_agent_config_pair() -> dict[str, AgentConfig]:
    agent_configs = _get_agent_configs()
    return {agent.name: agent for agent in agent_configs}


def get_agent_configs_str() -> str:
    agent_configs = _get_agent_configs()
    pair_list = [f"{agent.name}: {agent.description}" for agent in agent_configs]
    return "\n".join(pair_list)