---
name: create-pr-description
description: Creates comprehensive PR descriptions by analyzing commits, code changes, and gathering context from issues. Use when asked to write, generate, or create a PR description, or when preparing a pull request for review.
---

# PR Description Generator Skill

## Overview

This skill generates high-quality pull request descriptions for the RAGElo project by analyzing all changes, gathering context, and producing a human-readable summary. RAGElo is a Python library and CLI for evaluating RAG agents using Elo-based tournament ranking, published on PyPI.

## Output Format

The agent must produce:
1. **PR Title** - A concise, descriptive title (do NOT use conventional commits format like `feat(scope):` or `fix:`)
2. **PR Description** - Human-readable summary of changes
3. **Proposed Reviewers** - 1-2 people based on git history (excluding the PR author)

---

## Step 1: Gather All Changes

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

**IMPORTANT**: Do not just read commit messages. You MUST read the actual code changes to understand what was done. Commit messages may be incomplete or misleading.

---

## Step 2: Gather Context

### Read CLAUDE.md
Read `CLAUDE.md` at the repo root for project architecture, commands, and conventions.

### Check for related issues (MANDATORY)

**This step is required, not optional.** Keyword searches often miss relevant issues. Use multiple approaches:

#### 1. Browse recent open issues first (most reliable)
```bash
gh issue list --limit 50 --json number,title | jq -r '.[] | "\(.number): \(.title)"'
```

**Scan issue titles manually** - the tracking issue for your work may use different terminology than you'd search for.

#### 2. Then search by keywords
```bash
gh issue list --search "<keyword1>" --json number,title | jq -r '.[] | "\(.number): \(.title)"'
gh issue list --search "<keyword2>" --json number,title | jq -r '.[] | "\(.number): \(.title)"'
```

> **Note:** These searches default to open issues. If you need historical context on closed issues, add `--state closed` explicitly.

#### 3. Check the issue details
```bash
gh issue view <issue_number> --json title,body,labels,state | jq '.'
```

If changes solve or relate to an existing issue, mention it using `Closes #123` or `Relates to #123` syntax.

### Check documentation
Check `README.md` and `docs/` for relevant context and documentation that may need updating.

### Check affected code context
For significant changes, read surrounding code to understand:
- How things worked before
- Why the change improves the situation
- Impact on the factory/registry pattern, evaluator hierarchy, or public API

---

## Step 3: Determine PR Type and Structure

### For trivial changes (version bumps, typo fixes, simple config changes):
Keep it simple - one or two sentences explaining what changed.

### For feature implementations, use **What / Why / Impact** structure:
```markdown
## What
Brief description of what this PR does.

## Why
Explanation of the motivation and context.

## Impact
What parts of the system are affected, any behavioral changes.
```

### For refactoring or architectural changes, use **Background / Changes / Migration** structure:
```markdown
## Background
How things worked before and why that wasn't optimal.

## Changes
High-level description of the new approach.

## Migration
Any manual steps needed (if applicable).
```

### For bug fixes, use **Problem / Solution / Testing** structure:
```markdown
## Problem
Description of the bug and its symptoms.

## Solution
How this PR fixes it.

## Testing
How the fix was verified.
```

### For complex multi-part changes, use **Key Changes** with bullet points:
```markdown
## Summary
One paragraph overview.

## Key Changes
- **Feature A**: Brief description
- **Feature B**: Brief description
- **Refactoring**: Brief description
```

---

## Step 4: Include Diagrams When Appropriate

For large changes touching multiple components, include mermaid diagrams.

Use diagrams when they genuinely help understanding. Don't add them just for decoration.

---

## Step 5: Check for Special Conditions

### Public API / breaking changes
RAGElo is a published Python library. If the PR changes public-facing APIs (classes, functions, or CLI commands exposed to users), explicitly document:
1. Whether the change is backward compatible
2. Any deprecation warnings added
3. Impact on existing users

> ⚠️ **Breaking Change**: This PR modifies the public API. [Describe what changed and migration path]

### Security implications
Consider and document:
- LLM API key handling changes
- Data handling modifications (user queries, documents)
- New external integrations or dependencies

---

## Step 6: Find Reviewers

Use git blame to identify people who have recently modified the affected code:

```bash
git diff --name-only origin/master..HEAD | while read file; do
  git blame --line-porcelain "$file" 2>/dev/null | grep "^author " | sort | uniq -c | sort -rn
done | grep -v "$(git config user.name)" | sort | uniq -c | sort -rn | head -5
```

Select 1-2 reviewers who:
- Have significant contributions to the affected code
- Are not the PR author
- Have recent activity

---

## What NOT to Do

❌ Do not create lists of every file that changed with individual explanations
❌ Do not copy-paste commit messages as the description
❌ Do not include implementation details unless critical for understanding
❌ Do not write overly long descriptions for simple changes
❌ Do not skip reading the actual code changes
❌ Do not include the PR author as a reviewer
❌ Do not skip issue checking because "there probably isn't one"

## What TO Do

✅ Read all code changes thoroughly before writing
✅ Be concise - readers should understand the change quickly
✅ Focus on the "what" and "why", and to a lesser extent the "how" (unless the "how" is the point)
✅ Include diagrams for complex architectural changes
✅ Mention related issues or documentation
✅ Highlight breaking changes to the public API prominently
✅ Find reviewers who know the affected code
✅ Browse recent issues before writing - the tracking issue may exist under different terminology

---

## Example Output

```markdown
# PR Title
Add custom prompt support for retrieval evaluators

# PR Description

Closes #42.

## What
Adds a new `CustomPromptEvaluator` to the retrieval evaluator hierarchy, allowing users to supply their own Jinja2 prompt templates for document relevance evaluation.

## Why
Users with domain-specific evaluation criteria needed a way to customize the LLM prompt without forking the library or subclassing evaluators.

## Key Changes
- **CustomPromptEvaluator**: New evaluator registered via the factory pattern under `RetrievalEvaluatorTypes.CUSTOM_PROMPT`
- **CLI integration**: Added `custom-prompt` subcommand to `retrieval-evaluator` CLI
- **Templates**: Users can pass a Jinja2 template string or file path

## Impact
- No breaking changes to existing evaluators or public API
- New optional dependency on `jinja2` (already a transitive dependency)

---

# Proposed Reviewers
- @alice - Primary author of the evaluator hierarchy
- @bob - Recent contributor to CLI module
```