from dotenv import load_dotenv
import chainlit as cl
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context

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


def ready_my_workflow() -> tuple[CustomerService, Context]:
    memory = ChatMemoryBuffer(
        llm=llm,
        token_limit=5000
    )

    workflow = CustomerService(
        memory=memory,
        timeout=None,
        user_state=initialize_user_state()
    )

    context = Context(workflow)
    return workflow, context


def initialize_user_state() -> dict[str, str | None]:
    return {
        "name": None
    }


@cl.on_chat_start
async def start():
    workflow, ctx = ready_my_workflow()
    cl.user_session.set("workflow", workflow)
    cl.user_session.set("context", ctx)

    await cl.Message(
        author="assistant", content=GREETINGS
    ).send()


@cl.step(type="run", show_input=False)
async def on_progress(message: str):
    return message


@cl.on_message
async def main(message: cl.Message):
    workflow: CustomerService = cl.user_session.get("workflow")
    context = cl.user_session.get("context")
    msg = cl.Message(content="", author="assistant")
    user_msg = message.content
    handler = workflow.run(
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
