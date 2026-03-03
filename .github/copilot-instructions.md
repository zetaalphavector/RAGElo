# Copilot Instructions — RAGElo

## Project Overview

RAGElo is a Python library and CLI for evaluating RAG (Retrieval-Augmented Generation) agents using Elo-based tournament ranking. It evaluates both document retrieval quality and answer quality across multiple RAG pipeline variations.

## Commands

```bash
# Install for development
uv pip install -e '.[dev]'

# Run all tests (OpenAI integration tests are skipped by default)
pytest tests/

# Run with OpenAI integration tests (requires OPENAI_API_KEY)
pytest tests/ --runopenai

# Run a single test file or specific test
pytest tests/unit/test_experiment.py -v
pytest tests/unit/test_experiment.py::TestExperiment::test_method -v

# Lint and format (ruff rules: E, F, I; line-length: 119)
ruff check ragelo/
ruff format ragelo/

# Type checking (uses pydantic.mypy plugin)
mypy ragelo/

# Pre-commit hooks (ruff lint + format)
pre-commit run --all-files
```

## Architecture

### Factory + Registry Pattern

All major components use decorator-based factory registration. Enum types in `ragelo/types/types.py` define valid component names (`RetrievalEvaluatorTypes`, `AnswerEvaluatorTypes`, `LLMProviderTypes`, `AgentRankerTypes`).

```python
# Registration (on class definition)
@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.REASONER)
class ReasonerEvaluator(BaseRetrievalEvaluator): ...

# Instantiation (via factory functions)
evaluator = get_retrieval_evaluator("reasoner", llm_provider=provider)
provider = get_llm_provider("openai", api_key="...")
ranker = get_agent_ranker("elo")
```

When adding a new evaluator, LLM provider, or ranker:
1. Add the name to the corresponding `StrEnum` in `ragelo/types/types.py`.
2. Create a config class in `ragelo/types/configurations/`.
3. Implement the class and decorate it with the factory's `@register`.
4. The class is now accessible via the factory function and CLI.

### Evaluator Hierarchy

```
BaseEvaluator (ragelo/evaluators/base_evaluator.py) — async-first, abstract
├── BaseRetrievalEvaluator — evaluates document relevance (Query + Document → score)
│   Implementations: Reasoner, RDNAM, DomainExpert, FewShot, CustomPrompt
└── BaseAnswerEvaluator — evaluates answer quality (Query + AgentAnswer → score/winner)
    Implementations: Pairwise, ChatPairwise, CustomPairwise, DomainExpert, CustomPrompt
```

All evaluations are **async**: `evaluate_async()` is the core abstract method. `evaluate_experiment()` orchestrates bounded concurrent execution via `asyncio.wait()` with configurable `n_processes`.

### LLM Providers

`BaseLLMProvider` defines the interface: `call_async(input, response_schema) → LLMResponseType[T]`. Implementations: `OpenAIProvider` (structured output via Responses API), `OllamaProvider`. Providers return `LLMResponseType[T]` with `raw_answer` and `parsed_answer`.

### Data Model

- **Core types** (`ragelo/types/evaluables.py`): `Query`, `Document`, `AgentAnswer`, `PairwiseGame`
- **Results** (`ragelo/types/results.py`): `EvaluatorResult`, `RetrievalEvaluatorResult`, `AnswerEvaluatorResult`, `PairwiseGameEvaluatorResult`
- **LLM answer schemas** (`ragelo/types/answer_formats.py`): Pydantic models for structured LLM outputs
- **Configs** (`ragelo/types/configurations/`): Pydantic config classes per component, resolved via `get_config_class()` introspection

### Experiment Orchestration

`Experiment` (`ragelo/types/experiment.py`) is the central orchestrator: loads queries/documents/answers from CSV, manages evaluation state with caching to avoid redundant LLM calls, and persists to JSON/JSONL.

### Prompt Templating

Jinja2 templates for all LLM prompts. Available context variables: `{{ query.query }}`, `{{ document.text }}`, `{{ answer.text }}`, `{{ game.agent_a_answer.text }}`. Extra CSV columns become metadata: `{{ query.metadata.column_name }}`.

### CLI

Typer-based CLI (`ragelo/cli/`). Entry point: `ragelo = "ragelo.cli:app"`. Subcommands: `run-all`, `retrieval-evaluator <type>`, `answer-evaluator <type>`. CLI parameters are dynamically generated from Pydantic config `Field(description=...)`.

## Code Style (Mandatory)

The codebase must read as if one person wrote it. **Find the existing pattern first, then follow it.** Never introduce a novel approach when an established one exists.

### Python

**Imports — absolute and top-level only:**
- ✅ `from ragelo.types.configurations import BaseEvaluatorConfig`
- ❌ `from .configurations import BaseEvaluatorConfig` (relative imports)
- ❌ `from ..types import BaseEvaluatorConfig` (relative imports)
- ❌ Imports inside function bodies — always import at module top. Deferred imports hide dependencies and break tooling.
- Use `TYPE_CHECKING` blocks for forward references only.

**Private methods — single underscore:**
- ✅ `def _build_message(self, ...)` (single underscore — matches existing codebase)
- Public methods use no prefix: `def evaluate(self, ...)`
- This project uses single-underscore private methods throughout. Follow the existing convention.

**Comments and docstrings — minimal:**
- No docstrings for self-explanatory functions. If the function signature tells you everything, adding a docstring is noise.
- No comments that restate what the code does.
- Complex algorithms or non-obvious business rules may warrant brief comments.
- CLI `Field(description=...)` serves as documentation for user-facing parameters — keep those descriptive.

**Class design:**
- Single responsibility — one class, one job.
- Use composition over inheritance (except the evaluator hierarchy, which uses inheritance deliberately).
- Config objects are separate Pydantic models, not embedded logic.

**Pydantic conventions:**
- All config classes inherit from `BaseConfig` or its subclasses.
- Use `Field(default=..., description="...")` for all config fields.
- Use `@field_validator` for validation logic.
- Use `ConfigDict(arbitrary_types_allowed=True)` when needed (e.g., Jinja2 `Template` fields).

**Async patterns:**
- Core methods are `async` (`evaluate_async`, `call_async`).
- Sync wrappers use `call_async_fn()` from `ragelo/utils.py` — never call `asyncio.run()` directly.
- Bounded concurrency via `asyncio.wait()` with `n_processes`.

### The Chameleon Principle

Before writing any code, search for how the same problem is already solved in the codebase:
- **Naming**: Match existing conventions (e.g., factory functions are `get_<component>()`).
- **Error handling**: Use the established exception patterns.
- **Config patterns**: Follow the existing `BaseConfig` → `Component-specific Config` hierarchy.
- **File organization**: New evaluators go in their subsystem's directory with matching `__init__.py` exports.
- **Abstraction level**: If similar features use the factory pattern, don't bypass it.

If you can't find an existing pattern, that's a signal to ask before inventing one.

## Testing

### Approach

- Use `pytest` with `pytest-asyncio` and `pytest-mock`.
- `MockLLMProvider` in `tests/conftest.py` returns deterministic responses based on the requested Pydantic schema — use it for all evaluator tests.
- The `experiment` fixture loads test data from `tests/data/` (2 queries, 4 docs, 2 agents) — prefer using it over creating ad-hoc test data.
- OpenAI integration tests are gated with `@pytest.mark.requires_openai` — they only run when `--runopenai` is passed.

### Style

- Focus on testing critical behavior, not implementation details.
- Prefer a few high-signal cases over exhaustive/obvious ones. Cover the main success path plus 1–2 meaningful failure/edge paths.
- When several cases exercise the same behavior, combine them into a single representative test or use parametrize instead of one test per micro-variant.
- Class-based test organization: `TestRetrievalEvaluator`, `TestAnswerEvaluators`, etc.
- Never test private methods directly.
- Avoid monkeypatching — the codebase uses dependency injection (configs, LLM providers as constructor parameters). If you need to patch, the design may need refactoring.

## Self-Improvement

When you complete a task and realize you made a mistake that could have been avoided with better instructions in this file, **append a concise note** to the "Lessons Learned" section below. Keep entries specific and actionable.

## Lessons Learned

<!-- Add entries here when you discover something that would have prevented a mistake -->
