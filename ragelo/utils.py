from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine


def run(coroutine: Coroutine[Any, Any, Any]) -> Any:
    """
    Runs the given coroutine and returns its result.
    """
    return asyncio.run(coroutine)


def call_async_fn(fn: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any) -> Any:
    """
    Calls an asynchronous function, either in the current event loop
    or in a separate thread if no loop is running.
    """
    try:
        asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Execute the coroutine using the `run` function in a thread pool
            future = executor.submit(run, fn(*args, **kwargs))
            return future.result()
    except RuntimeError:
        # If no running loop is detected, run the coroutine directly
        return asyncio.run(fn(*args, **kwargs))
