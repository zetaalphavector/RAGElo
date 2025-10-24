from __future__ import annotations

import asyncio
import re
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


def get_placeholders_and_tags(template: Template) -> set[str]:
    source = getattr(template, "_ragelo_source", None)
    if source is None:
        return set()

    # Extract simple placeholders like {{ foo.bar }}
    placeholders = set(m.group(1) for m in re.finditer(r"{{\s*([a-zA-Z_][\w\.]*)\s*}}", source))

    # Extract variables mentioned inside Jinja tags, e.g. {% for x in items %}
    # This needs to handle whitespace and trim markers ({%- ... -%}).
    tag_bodies = re.findall(r"{%-?\s*(.*?)\s*-?%}", source, flags=re.DOTALL)

    # Identify all identifier-like tokens inside the tag body, keeping dotted paths.
    raw_idents: set[str] = set()
    for body in tag_bodies:
        for ident in re.findall(r"[a-zA-Z_][\w\.]*", body):
            raw_idents.add(ident)

    # Filter out Jinja keywords and control words; we only want variables.
    jinja_keywords = {
        # Control structures
        "for",
        "in",
        "if",
        "elif",
        "else",
        "endif",
        "endfor",
        "set",
        "endset",
        "with",
        "endwith",
        "block",
        "endblock",
        "filter",
        "endfilter",
        "macro",
        "endmacro",
        "call",
        "endcall",
        "autoescape",
        "endautoescape",
        "raw",
        "endraw",
        "do",
        # Import/include
        "include",
        "from",
        "import",
        "as",
        # Operators / tests
        "and",
        "or",
        "not",
        "is",
        # Literals
        "true",
        "false",
        "none",
        # Misc common words in templates
        "ignore",
        "missing",
    }
    tag_variables = {i for i in raw_idents if i.lower() not in jinja_keywords}

    return placeholders | tag_variables
