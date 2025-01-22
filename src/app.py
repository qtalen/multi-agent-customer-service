from dotenv import load_dotenv
import chainlit as cl
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.memory import ChatMemoryBuffer

from workflow import CustomerService, ProgressEvent

load_dotenv("../.env")

llm = OpenAILike(
    model="qwen-max-latest",
    is_chat_model=True,
    is_function_calling_model=True,
    temperature=0.35
)
Settings.llm = llm

GREETINGS = "Hello, what can I do for you?"


def ready_my_agent() -> CustomerService:
    memory = ChatMemoryBuffer(
        llm=llm,
        token_limit=5000
    )

    agent = CustomerService(
        memory=memory,
        timeout=None,
        user_state=initialize_user_state()
    )
    return agent


def initialize_user_state() -> dict[str, str | None]:
    return {
        "name": None
    }


@cl.on_chat_start
async def start():
    agent = ready_my_agent()
    user_state = initialize_user_state()
    cl.user_session.set("workflow", agent)
    cl.user_session.set("user_state", user_state)

    await cl.Message(
        author="assistant", content=GREETINGS
    ).send()


@cl.step(type="run", show_input=False)
async def on_progress(message: str):
    return message


@cl.on_message
async def main(message: cl.Message):
    agent: CustomerService = cl.user_session.get("workflow")
    context = cl.user_session.get("context")
    msg = cl.Message(content="", author="assistant")
    user_msg = message.content
    handler = agent.run(
        msg=user_msg,
        ctx=context
    )
    async for event in handler.stream_events():
        if isinstance(event, ProgressEvent):
            await on_progress(event.msg)

    await msg.send()
    result = await handler
    msg.content = result
    await msg.update()
    cl.user_session.set("context", handler.ctx)
