from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import chainlit as cl
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    AgentInput,
    AgentOutput,
    AgentStream
)
from llama_index.core.workflow import Context

from agents import concierge_agent, pre_sales_agent, post_sales_agent

load_dotenv("../../.env")

GREETINGS = "Hello, what can I do for you?"


def ready_my_workflow() -> tuple[AgentWorkflow, Context]:
    workflow = AgentWorkflow(
        agents=[concierge_agent, pre_sales_agent, post_sales_agent],
        root_agent=concierge_agent.name,
        handoff_output_prompt=(
            "handoff_result: Due to {reason}, the user's request has been passed to {to_agent}."
            "Please review the conversation history immediately and continue responding to the user's request."
        ),
        initial_state={
            "username": None
        }
    )
    ctx = Context(workflow=workflow)
    return workflow, ctx


@cl.on_chat_start
async def start():
    workflow, ctx = ready_my_workflow()
    cl.user_session.set("workflow", workflow)
    cl.user_session.set("context", ctx)

    await cl.Message(
        author="assistant", content=GREETINGS
    ).send()


@cl.step(name="thinking", type="run", show_input=False)
async def on_progress(message: str):
    return message


@cl.on_message
async def main(message: cl.Message):
    workflow: AgentWorkflow = cl.user_session.get("workflow")
    context: Context = cl.user_session.get("context")

    handler = workflow.run(
        user_msg=message.content,
        ctx=context
    )
    stream_msg = cl.Message(content="")
    async for event in handler.stream_events():
        if isinstance(event, AgentInput):
            print(f"========{event.current_agent_name}:=========>")
            print(event.input)
            print("=================<")
        if isinstance(event, AgentOutput) and event.response.content:
            print("<================>")
            print(f"{event.current_agent_name}: {event.response.content}")
            print("<================>")
        if isinstance(event, AgentStream):
            await stream_msg.stream_token(event.delta)
    await stream_msg.send()
