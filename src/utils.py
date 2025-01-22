from inspect import signature
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from typing import Any, Awaitable, Callable, Type, cast, override

from llama_index.core.tools import (
    FunctionTool,
    ToolOutput,
    ToolMetadata
)

from llama_index.core.workflow import (
    Context,
)

type AsyncCallable = Callable[..., Awaitable[Any]]


def create_schema_from_function(
        name: str,
        func: Callable[..., Any] | AsyncCallable,
        additional_fields: list[tuple[str, Type, Any] | tuple[str, Type]] | None = None
) -> Type[BaseModel]:
    """Create schema from function"""
    fields = {}
    params = signature(func).parameters
    for param_name in params:
        if param_name == "ctx":
            continue

        param_type = params[param_name].annotation
        param_default = params[param_name].default

        if param_type is params[param_name].empty:
            param_type = Any

        if param_default is params[param_name].empty:
            fields[param_name] = (param_type, FieldInfo())
        elif isinstance(param_default, FieldInfo):
            fields[param_name] = (param_type, param_default)
        else:
            fields[param_name] = (param_type, FieldInfo(default=param_default))

        additional_fields = additional_fields or []
        for field_info in additional_fields:
            if len(field_info) == 3:
                field_info = cast(tuple[str, Type, Any], field_info)
                field_name, field_type, field_default = field_info
                fields[field_name] = (field_type, FieldInfo(default=field_default))
            elif len(field_info) == 2:
                field_info = cast(tuple[str, Type], field_info)
                field_name, field_type = field_info
                fields[field_name] = (field_type, FieldInfo())
            else:
                raise ValueError(
                    f"Invalid additional field info: {field_info}."
                    "Must be a tuple of length 2 or 3"
                )
    return create_model(name, **fields)


class FunctionToolWithContext(FunctionTool):
    """
    A function tool that also includes passing in workflow context.
    Only override the call methods to include the context.
    """
    @classmethod
    @override
    def from_defaults(
            cls,
            fn: Callable[..., Any] | None = None,
            name: str | None = None,
            description: str | None = None,
            return_direct: bool = False,
            fn_schema: Type[BaseModel] | None = None,
            async_fn: AsyncCallable | None = None,
            tool_metadata: ToolMetadata | None = None
    ) -> "FunctionTool":
        if tool_metadata is None:
            fn_to_parse = fn or async_fn
            assert fn_to_parse is not None, "fn or async_fn must be provided."
            name = name or fn_to_parse.__name__
            docstring = fn_to_parse.__doc__

            signature_str = str(signature(fn_to_parse))
            signature_str = signature_str.replace(
                "ctx: llama_index.core.workflow.context.Context, ", ""
            )
            description = description or f"{name}{signature_str}\n{docstring}"
            if fn_schema is None:
                fn_schema = create_schema_from_function(
                    f"{name}", fn_to_parse, additional_fields=None
                )
            tool_metadata = ToolMetadata(
                name=name,
                description=description,
                fn_schema=fn_schema,
                return_direct=return_direct
            )
        return cls(fn=fn, metadata=tool_metadata, async_fn=async_fn)

    @override
    def call(self, ctx: Context, *args: Any, **kwargs: Any) -> ToolOutput:
        """Call."""
        tool_output = self._fn(ctx, *args, **kwargs)
        return ToolOutput(
            content=str(tool_output),
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output
        )

    @override
    async def acall(self, ctx: Context, *args: Any, **kwargs: Any) -> ToolOutput:
        """Call."""
        tool_output = await self._async_fn(ctx, *args, **kwargs)
        return ToolOutput(
            content=str(tool_output),
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output
        )