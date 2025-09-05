from __future__ import annotations

import asyncio
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from textwrap import dedent
from typing import Any, Coroutine

from jinja2 import Template
from tqdm import TqdmExperimentalWarning
from tqdm.auto import tqdm
from tqdm.rich import tqdm_rich


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


def get_pbar(
    total: int, use_rich: bool = True, ncols: int = 100, desc: str = "Evaluating", disable: bool = False
) -> tqdm | tqdm_rich:
    if use_rich:
        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
        return tqdm_rich(total=total, ncols=ncols, desc=desc, disable=disable)
    else:
        return tqdm(total=total, ncols=ncols, desc=desc, disable=disable)


def string_to_template(src: str) -> Template:
    clean_str = dedent(src).strip()
    template = Template(clean_str)
    template._ragelo_source = src  # type: ignore
    return template
