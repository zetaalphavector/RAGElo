# Roadmap: layering the evaluator stack

Work in flight, in dependency order. The through-line is separating three layers that already
exist informally, so that each crossing is a declared contract instead of a guess:

1. **Artifacts** — queries, reference answers, rubrics. Human-ownable, versioned, never
   regenerable by the judge that is scored against them.
2. **Judges** — read artifacts, write typed judgments onto evaluables. Identity is a name; a new
   judge should be cheap.
3. **Measures** — pure functions of (qrels, runs). No judge concept, as in the
   `trec_eval` → `pytrec_eval` → `ir_measures` lineage.

RAGElo has 2 and 3 half-separated and no layer 1. Every symptom below is a crossing that was
never specified, so callers renegotiate it with `isinstance`, `hasattr` and structural guessing.

## Method: characterize before refactoring

Steps 1-2 are behaviour-preserving refactors of code that is **currently untested**. Measured
with `pytest --cov` on 2026-08-03 (197 passed, 2 skipped):

| module | coverage |
|---|---|
| `ragelo/types/results.py` | 71% |
| `ragelo/types/query.py` | 70% |
| `ragelo/measures.py` | 65% |
| `ragelo/types/evaluator_utils.py` | 56% |

Uncovered lines land squarely on what steps 1-2 touch: 6 of the 7 `validate_answer_type`
validators, all of `Query.get_qrels`' branches, and every `score`/`reasoning`/`strigify_answer`
property. So each step starts by writing tests that pin the **current** behaviour, including the
behaviour we think is wrong — those get a separate, deliberate commit that changes the assertion
and says why.

Definition of done for every item: `pytest tests/`, `ruff check`, `ruff format --check`,
`mypy ragelo` all clean, and coverage of the touched module not lower than when the item started.

---

## Step 0 — characterization tests (blocks steps 1 and 2)

No production change. Pin the happy path and the pitfalls that steps 1-2 could silently break.
Not chasing coverage as a number: a test earns its place only where the behaviour is decided by
*our* logic and a wrong outcome would be silent.

- [x] `ragelo/measures.py`: `parse_measure` across `NAME`, `NAME@K`, `NAME(param=value)@K`, an
      unknown name, and `Measure` passthrough; coverage classification; subtopic in `iteration`.
      Owed regardless of this roadmap — it was shipped with the rubric coverage work untested.
      (65% → 86%.)
- [x] `AnswerEvaluatorResult` reloads a serialized `RubricPointwiseAnswerFormat` as that type,
      with its criteria intact (results.py:113-124, fallback branch uncovered). The union is
      untagged, so this works only because the strict type is attempted *first*; an adversarial
      review identified making that branch permissive as data corruption, and nothing currently
      fails if someone does.
- [x] `PairwiseGameEvaluatorResult` reloads a serialized `RubricAnswerFormat` as that type
      (results.py:163-174, same fallback shape, different union).
- [x] `Query.get_qrels` happy path with real values, and honouring `retrieval_evaluator_name`
      when several judges have scored the same document. Today's `test_get_qrels` asserts only
      `len(qrels) == 2`, so a step-2 protocol could change the values with nothing failing.
- [x] `Query.get_qrels` skips a document it cannot score instead of raising. Load-bearing for
      half-evaluated experiments; step 2 must not turn it into an exception.
- [x] `Query.get_qrels` **zeroes** a score below `relevance_threshold` rather than dropping the
      document. Subtle, and the obvious "fix" during a refactor is to drop it, which changes
      `Judged@k` without changing `nDCG@k`.

Deliberately **not** covered, and why:

- The 4 RDNAM result variants (results.py:245, 288, 311, 329). Their `answer` is a single
  concrete type, not a union, so the validator only coerces a dict — there is no branch to guess
  and pydantic decides the outcome.
- Field presence, type coercion and required-field errors anywhere. Pydantic's job.
- `get_qrels`' non-numeric-score guard, and the one-line `hasattr` wrappers behind
  `.score` / `.reasoning` / `.strigify_answer`. Reachable only via an answer type that does not
  exist; the tolerance test above already covers the shape that matters.

## Step 1 — discriminated unions on answer formats — DONE

**Symptom.** `EvaluatorResult.answer` is a union widened once per judge. Each widening costs a
hand-rolled `@model_validator(mode="before")` that infers the type structurally — try the strict
type, fall back — and new `hasattr` guards downstream. There are **7** such validators, **6**
`isinstance(self.answer, ...)` property guards, and **17** `hasattr`/`getattr` probes into
`answer`.

**Change.** Give each `EvaluationAnswer` subclass a `Literal` tag and declare the unions with
`Field(discriminator=...)`. Pydantic dispatches; most of the 7 validators become deletions.

**Why it is more than tidying.** Ordering-based discrimination is one careless edit from silent
corruption. A tag makes the failure impossible rather than merely discouraged. The
`RubricCoverageAnswerFormat` case shows the limit of the current approach: it declares every
field `RetrievalEvaluationAnswer` does, so no strict-first ordering can distinguish it, and it
currently relies on a key-presence check (results.py:62-80) that is the same fragility in nicer
clothing.

**Watch.** Serialized payloads gain a tag, so the read path must tolerate its absence for
already-saved experiments. Step 0's round-trip tests are the safety net; add one that loads a
tag-less payload written by the previous version.

## Step 2 — judgments declare their measurable projection — DONE

**Symptom.** Layer 3 reads layer 2's concrete classes by duck-typing. `Query.get_qrels` does
`getattr(evaluation, "score")`; `Experiment.get_rubric_qrels` does
`getattr(answer, "criteria_addressed", None)` — the latter written during the rubric coverage
work precisely because there was no contract to program against. Adding a judge with a new
payload today costs a union widening, a validator branch, and a new duck-typed reader in the
metrics layer.

**Change.** Two protocols on the answer formats, roughly `relevance() -> float | None` and
`subtopics() -> list[str]`. `get_qrels` and `get_rubric_qrels` collect from whatever implements
them. Deletes both `getattr` chains and, more importantly, means the next judge needs no change
in the metrics layer at all.

**Watch.** `get_qrels`' current tolerance for junk (warn and skip) is load-bearing for
half-evaluated experiments — preserve it, do not turn it into an exception. Step 0 pins it.

## Step 3 — rubric generation as a first-class artifact step

Part 2 of the rubric/nugget work, as scoped when part 1 landed.

- [ ] `RubricGenerator` with `from_documents` — today's `criteria_prompt` /
      `criteria_user_prompt`, extracted out of the two rubric evaluators.
- [ ] `RubricGenerator.from_reference_answer` — decompose a known-correct answer into criteria,
      for query sets that ship gold answers. This is the accountability artifact: an LLM must not
      be able to regenerate it from the query alone.
- [ ] Move rubric storage from the in-memory `criteria_cache` onto `query.rubric`, so a generated
      rubric is persisted, reviewable and shared by `rubric_coverage`, `rubric_pointwise` and
      `rubric_pairwise`. `config.rubrics` stays as a back-compat seed that writes into it.
- [ ] Rubric fingerprint on judgments, so editing one query's rubric re-judges that query and
      nothing else. Only matters once rubrics are generated and then pruned.
- [ ] Retires the duplication between the two rubric evaluators: verbatim identical `__init__`,
      twin `criteria_prompt`s, `_build_criteria` differing only by an optional argument.

## Step 4 — aggregation out of judges

**Symptom.** `rubric_pointwise_evaluator.py:201` computes `weighted_sum`, discards it in favour
of `average_score`, then line 247 multiplies the average back out
(`weighted_sum = answer_format.average_score * sum(...)`) to fold in the built-in criteria. An
aggregate is inverted to recover an intermediate that was thrown away.

**Root cause.** `average_score` is a *measure*, not a judgment. Aggregation lives inside a judge.

**Change.** `_process_answer` records per-criterion judgments only; aggregation moves to layer 3.
Evidence-recall and citation-quality become additional criteria rather than post-hoc score
corrections.

**Watch.** This changes what `average_score` means for saved experiments — the one item here
with a genuine migration concern. Sequence it after step 3 so the two rubric evaluators are
consolidated first; otherwise the same duplicated code gets refactored twice.

## Step 5 — artifact preparation as a phase

**Symptom.** Both rubric evaluators override `evaluate_async` — which the contributor guide says
to do "almost never" — because there is no phase for *prepare this query's artifacts* before
judging. The override also carries the score patching from step 4.

**Change.** Run artifact generators ahead of judging, populating `query.rubric`. Removes the
reason to override `evaluate_async` and retires `criteria_cache` entirely.

**Depends on** steps 3 and 4.

---

## Not doing, and why

- **Free-form evaluator names / rewriting `resolve_evaluator_result_type`.** Off the critical
  path now that `rubric_coverage` is registered. Revisit only if we want unregistered judges.
- **YAML prompt catalogues.** Every surveyed framework keeps prompts in data, and RAGElo is the
  outlier in keeping them in classes — but with twelve evaluators and one team the extension cost
  is not yet the bottleneck. Revisit if judges start multiplying per criterion.
- **Extracting a shared `BaseEvaluatorFactory`.** ~85 lines of real duplication between the two
  factories, but `registry: dict = {}` on a shared base is a shared-mutable-class-attribute trap
  and the precise return types are what `mypy ragelo` currently checks. Independent of this
  roadmap; do it on its own or not at all.


---

## Log

**Steps 1 and 2 landed together** (they touch the same two readers, and splitting them would have
left `get_qrels` duck-typing into freshly tagged formats).

Delivered: `answer_format` tags on all 10 answer formats, hidden from the JSON schema so the LLM is
never asked for them; the three storage unions declared with `Field(discriminator=...)`; the 7
bespoke validators reduced to one-line delegations to a single `_resolve_legacy_answer` shim;
`GradedJudgment` / `SubtopicJudgment` protocols, with `Query.get_qrels` and
`Experiment.get_rubric_qrels` consuming them instead of `getattr`.

Measured: duck-typed probes into `answer` 17 → 14 (the 13 remaining are the property wrappers on
result classes, which step 4 removes); `results.py` 338 → 311 lines while gaining a discriminator
and legacy shim. End-to-end coverage numbers unchanged.

Two things found on the way, both pre-existing:

- `Annotated[T, SkipJsonSchema]` passes the class rather than a marker instance, which makes
  `model_json_schema()` **raise** instead of hiding the field. All 14 uses were the broken form, so
  RDNAM's `reasoning` and the rubric coverage `score` were never actually hidden — and any provider
  path that builds a schema dict from those models would have crashed. Fixed to `SkipJsonSchema[T]`.
- Both bases derived the LLM request schema by unwrapping the *storage* union and taking the first
  member. So the union's member order silently decided what the LLM was asked to produce, and
  reordering it for legacy-matching purposes changed behaviour. Extracted as
  `default_answer_type`, which names the coupling and removes the duplicated unwrapping. **This is
  the layer confusion again** — the storage union should not be the source of the request schema.
  Worth its own step; folded into step 4's scope note rather than fixed here.

Ordering constraint discovered: the annotation's first member is the evaluator's own format (it
decides the request schema), while legacy structural matching needs most-constrained-first. These
are different orders, so the legacy order is passed explicitly to the shim rather than inferred
from the annotation.
