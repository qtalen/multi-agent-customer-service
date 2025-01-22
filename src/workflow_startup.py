from llama_index.core import Settings
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai_like import OpenAILike


class StreamChatEvent(Event):
    delta: str


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
    