from llama_index.core import Settings
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai_like import OpenAILike
from src.models import INDEXES, query_docs
from my_agent import MyFunctionAgent

llm = OpenAILike(
    model="qwen-plus-latest",
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


concierge_agent = FunctionAgent(
    name="ConciergeAgent",
    description="An agent to register user information, used to check if the user has already registered their title.",
    system_prompt=(
        "You are an assistant responsible for registering user information.\n"
        "You check from the state whether a user has registered their title.\n"
        "If it's not registered, you need to ask the user to register it.\n"
        "You are not allowed to make up a user's title.\n"
        "If the user has provided their information, you need to use the login tool to record this information.\n"
        "Once the information is recorded or if the state already has the user's information saved,\n"
        "you can pass control based on the user's intent to either the PreSalesAgent or the PostSalesAgent.\n"
    ),
    tools=[login],
    can_handoff_to=["PreSalesAgent", "PostSalesAgent"]
)


pre_sales_agent = FunctionAgent(
    name="PreSalesAgent",
    description="A pre-sales assistant helps answer customer questions about products and assists them in making purchasing decisions.",
    system_prompt=(
        "You are an assistant designed to answer users' questions about product information to help them make suitable decisions before purchasing.\n"
        "**When you receive a user's request, you will immediately review the conversation history and respond without making the user wait.**\n"
        "You must use the 'query_sku_info' tool to get the required information to answer the customer, and you cannot make up non-existent information.\n"
        "If the user is not seeking pre-purchase advice, you should hand over control to the ConciergeAgent or PostSalesAgent.\n"
    ),
    tools=[query_sku_info],
    can_handoff_to=["ConciergeAgent", "PostSalesAgent"]
)


post_sales_agent = FunctionAgent(
    name="PostSalesAgent",
    description="After-sales agent, used to answer user inquiries about product after-sales information, including product usage Q&A and after-sales policies.",
    system_prompt=(
        "You are an assistant designed to answer user inquiries about product after-sales information, including product usage Q&A and after-sales policies.\n"
        "**When you receive a user's request, you will immediately review the conversation history and respond without making the user wait.**\n"
        "You must use the 'query_terms_info' tool to get the required information to answer the customer, and you cannot make up non-existent information.\n"
        "If the user is not seeking after-sales or product usage-related advice, you should hand over control to the ConciergeAgent or PreSalesAgent.\n"
    ),
    tools=[query_terms_info],
    can_handoff_to=["ConciergeAgent", "PreSalesAgent"]
)
