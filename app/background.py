"""Utilities for supervising background asyncio tasks.

This ensures fire-and-forget jobs surface exceptions instead of
failing silently and prevents tasks from being garbage collected
before completion.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Executor
from functools import partial
from typing import Awaitable, Callable, Optional, Any, Set

logger = logging.getLogger(__name__)

# Track tasks so they are not garbage collected before finishing.
_background_tasks: Set[asyncio.Task[Any]] = set()


def spawn(coro: Awaitable[Any], *, name: Optional[str] = None,
          on_error: Optional[Callable[[BaseException], None]] = None) -> asyncio.Task[Any]:
    """Create and supervise a background task.

    Args:
        coro: Awaitable coroutine to run in the background.
        name: Optional name for the task (Python 3.8+ stores it on the task).
        on_error: Optional callback invoked if the task raises.
    """
    task = asyncio.create_task(coro, name=name)  # type: ignore[arg-type]
    _background_tasks.add(task)

    def _finished(t: asyncio.Task[Any]) -> None:
        _background_tasks.discard(t)
        try:
            t.result()
        except asyncio.CancelledError:
            logger.debug("Background task %s cancelled", name or t)
        except Exception as exc:  # noqa: BLE001
            if on_error:
                try:
                    on_error(exc)
                except Exception:  # noqa: BLE001
                    logger.exception("Error in on_error callback for task %s", name or t)
            logger.exception("Background task %s failed", name or t, exc_info=exc)

    task.add_done_callback(_finished)
    return task


__all__ = ["spawn"]


async def run_sync(func: Callable[..., Any], *args: Any, loop: Optional[asyncio.AbstractEventLoop] = None,
                   executor: Optional[Executor] = None, **kwargs: Any) -> Any:
    """Execute blocking code in the default executor and await the result."""
    event_loop = loop or asyncio.get_running_loop()
    bound = partial(func, *args, **kwargs)
    return await event_loop.run_in_executor(executor, bound)


__all__.append("run_sync")
