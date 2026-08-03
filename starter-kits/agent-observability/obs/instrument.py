"""Decorators and wrappers that add tracing to existing code.

Instrumentation should be something you add to a working system without restructuring
it. Everything here wraps a callable and preserves its signature and return value, so
removing a decorator returns you to the original behaviour exactly.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

from .pricing import TokenUsage
from .tracing import SpanKind, Tracer

F = TypeVar("F", bound=Callable[..., Any])


@runtime_checkable
class ModelResponse(Protocol):
    """Minimal shape a traced model call is expected to return.

    Kept structural rather than a base class so provider SDK objects, your own
    dataclasses, and test stubs all satisfy it without inheriting anything.
    """

    text: str
    usage: TokenUsage


def traced_tool(
    tracer: Tracer,
    name: str | None = None,
    *,
    capture_result: bool = True,
) -> Callable[[F], F]:
    """Wrap a tool function so each call becomes a TOOL span.

    Args:
        tracer: Tracer that owns the span.
        name: Span name; defaults to the function's ``__name__``.
        capture_result: Set False for tools that return large blobs you do not want
            copied into every trace file.
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.span(
                    span_name, kind=SpanKind.TOOL, inputs=_call_inputs(func, args, kwargs)
                ):
                    result = await func(*args, **kwargs)
                    if capture_result:
                        tracer.record_output(result)
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.span(
                span_name, kind=SpanKind.TOOL, inputs=_call_inputs(func, args, kwargs)
            ):
                result = func(*args, **kwargs)
                if capture_result:
                    tracer.record_output(result)
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


def traced_model_call(
    tracer: Tracer,
    model: str,
    name: str | None = None,
) -> Callable[[F], F]:
    """Wrap a model-calling function so each call becomes a MODEL span.

    The wrapped function should return an object exposing ``text`` and ``usage``
    (see :class:`ModelResponse`). Usage is read off the response rather than estimated,
    because the provider's own count is the number you will be billed on.
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.span(
                    span_name,
                    kind=SpanKind.MODEL,
                    model=model,
                    inputs=_call_inputs(func, args, kwargs),
                ):
                    response = await func(*args, **kwargs)
                    _record_response(tracer, response)
                    return response

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.span(
                span_name,
                kind=SpanKind.MODEL,
                model=model,
                inputs=_call_inputs(func, args, kwargs),
            ):
                response = func(*args, **kwargs)
                _record_response(tracer, response)
                return response

        return wrapper  # type: ignore[return-value]

    return decorator


def _record_response(tracer: Tracer, response: Any) -> None:
    """Pull text and usage off a response object without assuming its concrete type."""
    text = getattr(response, "text", None)
    if text is not None:
        tracer.record_output(text)
    usage = getattr(response, "usage", None)
    if isinstance(usage, TokenUsage):
        tracer.record_usage(usage)


def _call_inputs(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Best-effort mapping of a call's arguments to parameter names.

    Binding can fail for wrapped or variadic callables. Tracing a call must never break
    the call, so failure degrades to the raw positional/keyword values.
    """
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("self", None)
        return arguments
    except (TypeError, ValueError):
        return {"args": list(args), "kwargs": kwargs}


class TracedModelClient:
    """Wraps any model client that exposes ``complete(prompt, **kwargs)``.

    Use this when you cannot decorate the call site, e.g. a client object handed to you
    by a framework. It traces without the framework knowing anything about tracing.
    """

    def __init__(self, inner: Any, tracer: Tracer, model: str) -> None:
        self._inner = inner
        self._tracer = tracer
        self._model = model

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        """Call the wrapped client inside a MODEL span."""
        with self._tracer.span(
            f"model.{self._model}",
            kind=SpanKind.MODEL,
            model=self._model,
            inputs={"prompt": prompt, **kwargs},
        ):
            response = self._inner.complete(prompt, **kwargs)
            _record_response(self._tracer, response)
            return response

    def __getattr__(self, item: str) -> Any:
        """Forward everything else to the wrapped client."""
        return getattr(self._inner, item)
