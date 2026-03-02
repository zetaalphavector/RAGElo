---
name: review-pr
description: Reviews pull requests for code quality, pattern conformance, architecture alignment, security, and completeness. Use when asked to review a PR, validate code changes, or check if a PR is ready for merge.
---

# PR Review Skill

## Overview

RAGElo is a Python library and CLI for evaluating RAG (Retrieval-Augmented Generation) agents using Elo-based tournament ranking. This skill reviews pull requests to ensure they:
- **Actually implement what the PR description claims**
- Follow existing codebase patterns (the Chameleon Principle)
- Match the code style of the existing codebase
- Align with RAGElo's established architecture (Factory+Registry, async evaluators, Pydantic configs)
- Address security concerns properly
- Introduce no breaking changes to the public API (or have proper migration paths)
- Relate to existing GitHub issues appropriately

### The Chameleon Principle

> The codebase should feel like it was architected by one mind, not assembled by mercenaries.

Every PR must introduce changes that blend seamlessly with existing patterns. When reviewing:
- **Find the existing pattern first** - Before accepting any change, search for how similar problems are already solved
- **Reject foreign patterns** - If the PR introduces a pattern that doesn't exist elsewhere, flag it ruthlessly
- **Suggest the existing way** - Always recommend the established approach over novel solutions
- **Be ruthless** - Pattern violations are not style nits; they are architectural debt

This principle applies to **everything**: architecture, code organization, naming conventions, testing approaches, and code style. AI agents are notorious for ignoring existing patterns and producing the same generic slop everywhere. The reviewer must catch this.

## Output Format

The agent must produce:
1. **Summary** - Brief overview of what the PR does
2. **Implementation Verification** - Does the code actually do what the PR claims?
3. **Breaking Changes** - Any backward-incompatible changes to the public API and migration paths
4. **Pattern & Style Conformance** - Does the PR follow existing codebase patterns and style?
5. **Architecture Alignment** - Is the PR consistent with RAGElo's Factory+Registry architecture?
6. **Code Quality & Testing** - Test coverage, test quality, code organization
7. **Issue Linkage** - Related GitHub issues
8. **Security Concerns** - Issues identified against security requirements
9. **Recommendations** - Required changes, suggestions, and approval status

---

## Step 1: Gather PR Context

### Get the PR description
```bash
gh pr view --json title,body,number | jq '.'
```

The PR description tells you what the author **claims** the PR does. Your job is to verify this claim against the actual code.

### Get the list of commits
```bash
git log origin/master..HEAD --oneline
```

### Get the full diff
```bash
git diff origin/master..HEAD
```

### Get list of changed files
```bash
git diff --name-only origin/master..HEAD
```

**CRITICAL**: Do not rely on commit messages or PR descriptions alone. You MUST read the actual code changes to understand what was done. The PR description may be incomplete, misleading, or outright wrong.

---

## Step 2: Verify Implementation Matches PR Description

This is a **primary concern**. Compare what the PR description claims against what the code actually does:

- Does the code implement all features/fixes mentioned in the description?
- Does the code do anything NOT mentioned in the description?
- Are there discrepancies between stated behavior and actual behavior?

If there are discrepancies:

> ⚠️ **Implementation Mismatch**
> 
> **PR claims**: [what the description says]
> 
> **Code actually does**: [what you found in the diff]
> 
> **Missing**: [features claimed but not implemented]
> 
> **Undocumented**: [changes made but not mentioned]

---

## Step 3: Check for Breaking Changes

RAGElo is a published library (`pip install ragelo`). **We never introduce breaking changes without proper migration paths.**

### Identify breaking changes in:
- Public Python API (functions in `ragelo/__init__.py`, factory functions like `get_retrieval_evaluator()`, `get_llm_provider()`, `get_agent_ranker()`)
- CLI interface (Typer commands in `ragelo/cli/`)
- Pydantic configuration classes (removed fields, changed types, changed defaults)
- Data model types (`ragelo/types/evaluables.py`, `ragelo/types/results.py`)
- Experiment JSON/JSONL serialization format
- Enum values in `ragelo/types/types.py` (removing or renaming enum members)

For each breaking change:

> 🚨 **Breaking Change Identified**
> 
> **What breaks**: [Describe the incompatibility]
> 
> **Affected consumers**: [Who/what depends on this]
> 
> **Migration path provided**: [Yes/No - describe if present]
> 
> **Required action**: [How to make this backward compatible, or document migration]

---

## Step 4: Check for Related GitHub Issues

### Browse recent open issues
```bash
gh issue list --limit 50 --json number,title | jq -r '.[] | "\(.number): \(.title)"'
```

### Search by keywords from the PR
```bash
gh issue list --search "<keyword>" --json number,title | jq -r '.[] | "\(.number): \(.title)"'
```

If there's an obvious tracking issue that should be linked but isn't in the PR description:

> 💡 **Suggested Issue Link**: This PR appears to address issue #123 ([issue title]). Consider adding `Closes #123` or `Relates to #123` to the PR description.

---

## Step 5: Pattern Conformance Analysis (Be Ruthless)

### The cardinal rule: Find the existing pattern FIRST

Before evaluating ANY code change, actively search the codebase for how similar problems are already solved.

### Pattern discovery process

1. **Identify the problem category**
   - New evaluator? → Check existing evaluators in `ragelo/evaluators/retrieval_evaluators/` and `ragelo/evaluators/answer_evaluators/`
   - New LLM provider? → Check `ragelo/llm_providers/openai_client.py` and `ragelo/llm_providers/ollama_client.py`
   - Configuration? → Check existing configs in `ragelo/types/configurations/`
   - New data type? → Check `ragelo/types/evaluables.py` and `ragelo/types/results.py`
   - Prompt template? → Check existing Jinja2 templates in evaluator classes

2. **Search for analogous implementations**
   Use `semantic_search` or `grep_search` to find 3+ examples of the same pattern to confirm it's established.

3. **Document the existing pattern**
   - Where is it used?
   - What are its characteristics?
   - Why was it chosen?

### RAGElo-specific pattern conformance checklist

**Factory + Registry pattern:**
- [ ] New components registered via decorator: `@Factory.register(EnumType.NAME)`
- [ ] Enum value added to `ragelo/types/types.py` (`RetrievalEvaluatorTypes`, `AnswerEvaluatorTypes`, `LLMProviderTypes`, `AgentRankerTypes`)
- [ ] Factory `create()` method works with the new component
- [ ] Convenience function updated (e.g., `get_retrieval_evaluator()`, `get_llm_provider()`)

**Evaluator pattern:**
- [ ] Inherits from `BaseRetrievalEvaluator` or `BaseAnswerEvaluator`
- [ ] Implements `evaluate_async()` as the core abstract method
- [ ] Uses `self.llm_provider` for LLM calls — does not instantiate its own provider
- [ ] Returns proper result type (`RetrievalEvaluatorResult`, `AnswerEvaluatorResult`, `PairwiseGameEvaluatorResult`)
- [ ] Uses Jinja2 `Template` for prompt formatting (not f-strings or `.format()`)
- [ ] Has a corresponding Pydantic config class in `ragelo/types/configurations/`

**Configuration pattern:**
- [ ] Config class inherits from appropriate base (`BaseEvaluatorConfig`, `BaseRetrievalEvaluatorConfig`, `BaseAnswerEvaluatorConfig`, `LLMProviderConfig`)
- [ ] Uses Pydantic `BaseModel` conventions (type annotations, defaults, validators)
- [ ] Config class is discoverable via `get_config_class()` introspection

**LLM Provider pattern:**
- [ ] Inherits from `BaseLLMProvider`
- [ ] Implements `call_async()` returning `LLMResponseType[T_Schema]`
- [ ] Accepts `LLMProviderConfig` (or subclass) in constructor
- [ ] Registered via `@LLMProviderFactory.register(LLMProviderTypes.NAME)`
- [ ] Supports structured output via `response_schema` parameter

**Data model pattern:**
- [ ] Core types (`Query`, `Document`, `AgentAnswer`, `PairwiseGame`) used as-is — not subclassed unnecessarily
- [ ] Result types inherit from `EvaluatorResult`
- [ ] LLM answer schemas in `ragelo/types/answer_formats.py` use Pydantic `BaseModel`
- [ ] Extra CSV columns → metadata dict pattern respected

**Prompt templating pattern:**
- [ ] Jinja2 `Template` objects, not raw string formatting
- [ ] Template variables match established names: `{{ query.query }}`, `{{ document.text }}`, `{{ answer.text }}`, `{{ game.agent_a_answer.text }}`
- [ ] Metadata accessed via `{{ query.metadata.column_name }}`

**CLI pattern:**
- [ ] Uses Typer (`ragelo/cli/`)
- [ ] CLI parameters dynamically generated from Pydantic config classes
- [ ] Follows existing subcommand structure

### Red flags to ruthlessly reject

🚫 **Novel patterns when established ones exist**
- "Let's use a different approach here" → NO. Use the existing approach.

🚫 **Inconsistent naming**
- `user_id` in one place, `userId` in another → NO. Match existing convention (snake_case throughout).

🚫 **Bypassing the Factory+Registry pattern**
- Directly instantiating evaluators instead of using `get_retrieval_evaluator()` → NO. Use the factory.

🚫 **New dependencies for solved problems**
- "Let's add library X" when existing code solves it → NO. Use existing solution.

🚫 **Different error handling**
- Custom exception hierarchies that don't match existing → NO. Use established patterns.

🚫 **f-strings or `.format()` for LLM prompts**
- All prompts must use Jinja2 `Template` objects → NO exceptions.

🚫 **Synchronous-only evaluator implementations**
- All evaluators must implement `evaluate_async()` → NO synchronous-only variants.

### When flagging pattern violations

Always show the existing pattern as evidence:

> **Pattern Violation**: PR uses f-string for prompt formatting
> 
> **Existing Pattern**: Codebase uses Jinja2 `Template` for all LLM prompts
> 
> **Evidence**: Found in `ragelo/evaluators/retrieval_evaluators/reasoner_evaluator.py`, `ragelo/evaluators/answer_evaluators/pairwise_evaluator.py`
> 
> **Required Fix**: Convert to Jinja2 `Template`

---

## Step 6: Code Style Conformance (Detect AI Slop)

AI coding agents are notorious for producing generic, pattern-ignorant code. The codebase must read as if a single person wrote it. Flag these common AI slop indicators:

### Linting & Formatting

RAGElo uses **ruff** for linting and formatting:
- Rules: `E`, `F`, `I` (errors, pyflakes, isort)
- Line length: **119** characters
- Type checking: **mypy** with `pydantic.mypy` plugin

Verify with:
```bash
ruff check ragelo/
ruff format --check ragelo/
mypy ragelo/
```

### Class Design

**Single Responsibility Principle:**
- [ ] Each class has one clear responsibility
- [ ] Classes are not "god objects" doing everything

**Dependency Injection:**
- [ ] `BaseLLMProvider` is injected into evaluators, not instantiated inside them
- [ ] Evaluators receive config objects, not raw kwargs

### Imports

**Absolute imports from `ragelo` package:**
- ✅ `from ragelo.types.evaluables import Query`
- ❌ `from .evaluables import Query`
- ❌ `from ..types.evaluables import Query`

**Note:** `from __future__ import annotations` is used throughout the codebase for forward references.

### Comments and Docstrings

**Minimal, meaningful documentation:**
- [ ] No useless comments stating the obvious
- [ ] Docstrings only when function signature is not self-explanatory

> **Style Violation**: Excessive comments/docstrings
> ```python
> def get_evaluator(name: str) -> BaseEvaluator:
>     """Get an evaluator by name.
>     
>     Args:
>         name: The name of the evaluator
>     
>     Returns:
>         The evaluator instance
>     """
> ```
> 
> **Required Fix**: Remove docstring - the function signature is self-explanatory

### Async Conventions

- [ ] Core evaluation logic is in `evaluate_async()` methods
- [ ] `call_async_fn()` utility used for sync→async bridge (from `ragelo.utils`)
- [ ] `asyncio.wait()` used for bounded concurrent execution in `evaluate_experiment()`
- [ ] No mixing of sync and async patterns within the same flow

---

## Step 7: Testing Quality

### Testing Patterns in RAGElo

RAGElo uses `pytest` with `pytest-asyncio` and `pytest-mock`.

**Key testing conventions:**
- `MockLLMProvider` in `tests/conftest.py` returns deterministic responses based on the requested Pydantic schema
- `experiment` fixture loads test data from `tests/data/` (2 queries, 4 docs, 2 agents)
- OpenAI integration tests gated with `@pytest.mark.requires_openai` (skipped unless `--runopenai`)
- Tests organized under `tests/unit/` and `tests/cli/`

### Test Quality Checklist

- [ ] Uses `MockLLMProvider` or similar fixture — not hand-rolled mocks that mirror implementation
- [ ] Tests the public interface (factory functions, `evaluate_experiment()`) not internal methods
- [ ] Uses existing fixtures (`experiment`, `llm_provider_config`, etc.)
- [ ] OpenAI-dependent tests properly gated with `@pytest.mark.requires_openai`
- [ ] Async tests use `pytest-asyncio` conventions

**Test Coverage Priority:**
1. Happy paths (highest priority)
2. Important edge cases (malformed input, missing fields)
3. Error handling paths
4. **NOT**: Trivial cases that add no value

**AI Slop Test Indicators:**
- [ ] Testing obvious getters/setters
- [ ] Excessive mocking that mirrors implementation details
- [ ] Tests that break when implementation changes (brittle)
- [ ] Not using existing fixtures and patterns from `conftest.py`

### Running Tests

```bash
# All tests (OpenAI skipped)
pytest tests/

# With OpenAI integration tests
pytest tests/ --runopenai

# Single file
pytest tests/unit/test_experiment.py -v
```

---

## Step 8: Architecture Alignment

### RAGElo Architecture Overview

Verify the PR respects these architectural patterns:

**Factory + Registry Pattern:**
- All major components use decorator-based factory registration
- Enum types in `ragelo/types/types.py` define valid component names
- Registration: `@Factory.register(EnumType.NAME)` on the class
- Instantiation: via factory functions (`get_retrieval_evaluator()`, `get_llm_provider()`, `get_agent_ranker()`)

**Evaluator Hierarchy:**
```
BaseEvaluator (ragelo/evaluators/base_evaluator.py) — async-first, abstract
├── BaseRetrievalEvaluator — evaluates document relevance (Query + Document -> score)
│   Implementations: Reasoner, RDNAM, DomainExpert, FewShot, CustomPrompt
└── BaseAnswerEvaluator — evaluates answer quality (Query + AgentAnswer -> score/winner)
    Implementations: Pairwise, ChatPairwise, CustomPairwise, DomainExpert, CustomPrompt
```

**LLM Providers:**
- `BaseLLMProvider` → `call_async(input, response_schema)` → `LLMResponseType[T]`
- Implementations: `OpenAIProvider`, `OllamaProvider`

**Data Model Layering:**
- Core evaluables in `ragelo/types/evaluables.py`: `Query`, `Document`, `AgentAnswer`, `PairwiseGame`
- Results in `ragelo/types/results.py`: `EvaluatorResult` and subclasses
- LLM answer schemas in `ragelo/types/answer_formats.py`: Pydantic models for structured LLM output
- Configs in `ragelo/types/configurations/`: one config class per component

**Experiment Orchestration:**
- `Experiment` class in `ragelo/types/experiment.py` is the central orchestrator
- Loads from CSV, manages evaluation state, caches to avoid redundant LLM calls, persists to JSON/JSONL

If architecture violations are found:

> ⚠️ **Architecture Violation**: [Describe the violation]
> 
> **Expected pattern**: [Explain the correct RAGElo pattern with file references]
> 
> **Required Fix**: [Specific remediation]

---

## Step 9: Security Review

Review security concerns relevant to a library that makes LLM API calls. Do NOT output a checklist. Instead, for each concern identified, use this format:

### Security Requirements to Verify

- No hardcoded API keys, secrets, or credentials (use environment variables)
- API keys handled via `SecretStr` or similar — not logged or printed
- Jinja2 templates not vulnerable to template injection from user-supplied metadata
- No arbitrary code execution from user-supplied prompts or configuration
- Dependencies properly declared in `pyproject.toml` with version constraints
- Cached results (JSON/JSONL files) don't leak sensitive information

### Output Format for Security Issues

For each security concern:

> 🔴 **Security Concern**: [Brief title]
> 
> **Issue**: [What the code does wrong or fails to do]
> 
> **Risk**: [Potential exploit, impact, or vulnerability]
> 
> **Required Fix**: [Specific remediation steps]

If no security concerns are identified, state: "No security concerns identified in this review."

---

## Step 10: Generate Review Summary

Compile findings into a structured review:

```markdown
# PR Review: [PR Title]

## Summary
[1-2 sentence overview of what the PR does]

## Implementation Verification
[✅ Matches PR description / ⚠️ Discrepancies found]
[List any mismatches between description and code]

## Breaking Changes
[🚨 Breaking changes found / ✅ No breaking changes]
[List any breaking changes with migration path assessment]

## Pattern & Style Conformance
[✅ Passes / ⚠️ Issues Found]
[List any violations with required fixes]

## Architecture Alignment
[✅ Aligned / ⚠️ Violations Found]
[List any architecture violations]

## Code Quality & Testing
[✅ Good / ⚠️ Issues Found]
[List any testing or code quality issues]

## Issue Linkage
- **Related Issues**: [List or "None identified"]

## Security Concerns
[List concerns in Security Concern format, or "No security concerns identified"]

## Recommendations

### Required Changes (Blocking)
1. [Critical issues that must be fixed]

### Suggested Improvements (Non-blocking)
1. [Nice-to-have improvements]

## Approval Status
[✅ Approved / ⚠️ Approved with comments / 🚫 Changes requested]
```

---

## What NOT to Do

❌ Do not approve without reading the actual code changes
❌ Do not trust PR descriptions without verification
❌ Do not skip pattern and style conformance checking
❌ Do not accept novel patterns when established ones exist
❌ Do not treat pattern violations as style preferences
❌ Do not ignore security concerns (especially API key handling)
❌ Do not allow breaking changes to the public API without migration paths
❌ Do not let the codebase feel like it was built by mercenaries
❌ Do not accept AI slop (generic, pattern-ignorant code)
❌ Do not accept tests that don't use existing fixtures from `conftest.py`
❌ Do not accept f-strings or `.format()` for LLM prompt construction
❌ Do not accept synchronous-only evaluator implementations
❌ Do not accept components that bypass the Factory+Registry pattern

## What TO Do

✅ Verify code actually implements what the PR description claims
✅ Flag all breaking changes to the public Python API and CLI
✅ **Search for existing patterns BEFORE evaluating changes** - This is mandatory
✅ **Cite specific file:line references when flagging violations**
✅ Verify code style matches existing codebase (ruff rules, 119 char lines, snake_case)
✅ Verify new components follow Factory+Registry pattern with enum types
✅ Verify evaluators are async-first and use Jinja2 templates
✅ Verify testing follows existing patterns (`MockLLMProvider`, fixtures, `@pytest.mark.requires_openai`)
✅ Check for related GitHub issues
✅ Review API key handling and credential security
✅ **Ruthlessly reject foreign patterns** - Suggest the existing way instead
✅ **Be a chameleon** - Changes must blend seamlessly with existing code
✅ Flag monolith page files and demand decomposition into `_lib/` and `_components/`
✅ Flag duplicated UI patterns and utility functions — demand shared components/modules
✅ Flag complex inline JSX callbacks — demand named handler functions
✅ Provide specific, actionable feedback
✅ Distinguish between blocking and non-blocking issues
