from llama_index.core import Settings
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai_like import OpenAILike
from src.models import INDEXES, query_docs
from my_agent import MyFunctionAgent

llm = OpenAILike(
    model="qwen-max-latest",
    is_chat_model=True,
    is_function_calling_model=True,
    temperature=0.01
)
Settings.llm = llm


async def login(ctx: Context, username: str) -> str:
    """
    Used to register user information, here it means the user's name.
    :param username: The user's name.
    :return:
    """
    current_state = await ctx.get("state", {})
    current_state["username"] = username
    await ctx.set("state", current_state)
    return "Username has been recorded!"


async def query_sku_info(ctx: Context, query: str) -> str:
    """
    A tool used to get the description of a product.
    :param query: The text of the user's query.
    :return:
    """
    sku_info = await query_docs(INDEXES["SKUS"], query)
    return sku_info


async def query_terms_info(ctx: Context, query: str) -> str:
    """
    A tool used to get product usage Q&A and post-sales terms.
    :param query: The text of the user's query.
    :return:
    """
    terms_info = await query_docs(INDEXES["TERMS"], query)
    return terms_info


concierge_agent = MyFunctionAgent(
    name="ConciergeAgent",
    description="An agent to register user information, used to check if the user has already registered their title.",
    system_prompt=(
        "You are an assistant responsible for recording user information."
        "You check from the state whether the user has provided their title or not."
        "If they haven't, you should ask the user to provide it."
        "You cannot make up the user's title."
        "If the user has already provided their information, you should use the login tool to record this information."
    ),
    tools=[login],
    can_handoff_to=["PreSalesAgent", "PostSalesAgent"]
)


pre_sales_agent = MyFunctionAgent(
    name="PreSalesAgent",
    description="A pre-sales assistant helps answer customer questions about products and assists them in making purchasing decisions.",
    system_prompt=(
        "You are an assistant designed to answer users' questions about product information to help them make the right decision before purchasing."
        "You must use the query_sku_info tool to get the necessary information to answer the user and cannot make up information that doesn't exist."
        "If the user is not asking pre-purchase questions, you should transfer control to the ConciergeAgent or PostSalesAgent."
    ),
    tools=[query_sku_info],
    can_handoff_to=["ConciergeAgent", "PostSalesAgent"]
)


post_sales_agent = MyFunctionAgent(
    name="PostSalesAgent",
    description="After-sales agent, used to answer user inquiries about product after-sales information, including product usage Q&A and after-sales policies.",
    system_prompt=(
        "You are an assistant responsible for answering users' questions about product after-sales information, including product usage Q&A and after-sales policies."
        "You must use the query_terms_info tool to get the necessary information to answer the user and cannot make up information that doesn't exist."
        "If the user is not asking after-sales or product usage-related questions, you should transfer control to the ConciergeAgent or PreSalesAgent."
    ),
    tools=[query_terms_info],
    can_handoff_to=["ConciergeAgent", "PreSalesAgent"]
)
