# Benchmarks

`run_llmjudge.py` and `run_llmjudge_pairwise.py` judge passages and compare the judgments with the human
labels. Each table prints how many pairs failed, and the agreement is computed over the pairs every row judged.

## Usage

Every run calls a paid API. Set `OPENAI_API_KEY` for `--provider openai` (the default of `run_llmjudge`), or
`AI_GATEWAY_API_KEY` for `vercel` and `vercel-jev`. The correlations come from SciPy, which the dev group
installs and the `benchmarks` extra declares (`pip install 'ragelo[benchmarks]'`). Run from the repository root:

```sh
# Retrieval evaluators on a label-stratified sample, with the cost per 1,000 pairs
uv run --python 3.13 python -m benchmarks.run_llmjudge --model gpt-5.6-luna --n-pairs 500 \
    --evaluator reasoner --evaluator rdnam --price gpt-5.6-luna=0.20,1.20,0.02

# The same on the TREC 2024 RAG qrels, judged by Jev
uv run --python 3.13 python -m benchmarks.run_llmjudge --dataset trec_rag24 --n-pairs 2000 \
    --provider vercel-jev --model typesafe-ai/jev --evaluator jev_boolean --evaluator jev_rdnam

# Pairwise answer evaluators: which of two passages of a query did the assessors grade higher
uv run --python 3.13 python -m benchmarks.run_llmjudge_pairwise --model typesafe-ai/jev
```

| Option | Meaning |
|---|---|
| `--model`, `--evaluator` | Repeat either to add rows. The retrieval variants are the keys of `EVALUATORS` in `run_llmjudge.py`. Always pass `--evaluator` to `run_llmjudge`: its default is every variant, and the `jev_*` ones raise on any provider but `vercel-jev`. |
| `--n-pairs`, `--seed` | Judge a sample that keeps the label distribution. Without `--n-pairs` the whole split is judged. |
| `--split` | `dev` or `test` for `llmjudge`. `trec_rag24` has `test` only. |
| `--price` | `MODEL=INPUT,OUTPUT,CACHED` in USD per million tokens. Jev's price is built in; for any other model the table shows tokens and no cost without it. |
| `--n-processes` | Parallel evaluations, 16 by default. At 32 the Jev gateway stalled. |
| `--tag` | Suffix of the experiment names, to judge again without reusing a cache. |

Judgments are cached per evaluator, model, sample and tag under `benchmarks/results/<dataset>/`, and the data
under `benchmarks/data/<dataset>/`. Both are ignored by git. Running the same command again judges only the
pairs that are missing or failed, so that is also how to retry failures, and a fully cached run reprints the
table without calling the API.

The table is ranked by the Spearman of the rounded labels, with a 95% interval that resamples queries.
`failed` counts the evaluations that returned no judgment. Tokens and cost cover the judged pairs only.
`evaluations / s` is the median over the timed runs at the current `--n-processes`.

To benchmark another configuration, add an entry to `EVALUATORS`. To add a dataset, write a module with
`download(data_dir)` and `load(data_dir, split)` that returns an `LLMJudgeData`, and list it in `DATASETS`.

## System ranking on the TREC 2024 RAG answers

`run_trec_rag24_answers.py` plays an Elo tournament between the systems of the TREC 2024 RAG track and compares
the ranking with the one from NIST's nugget judgments of the same answers.

```sh
uv run --python 3.13 python -m benchmarks.run_trec_rag24_answers --evaluator jev_pairwise --evaluator jev_rubric_pairwise
uv run --python 3.13 python -m benchmarks.run_trec_rag24_answers --provider openai --model gpt-5.6-luna \
    --evaluator pairwise --evaluator rubric_pairwise --price gpt-5.6-luna=0.20,1.20,0.02
```

NIST's [final nugget assessments](https://trec.nist.gov/data/rag2024.html) hold 2,471 full answers from 45 runs on
56 topics, no password needed. A nugget is an atomic fact a good answer should contain, marked vital or okay,
and every answer is labelled support, partial_support or not_support on each nugget of its topic. That is the
shape of a RAGElo rubric: `trec_rag24_answers.rubrics` turns a topic's nuggets into its criteria, weighted 1
and 0.5 as in the track, and the evaluators with "rubric" in their name judge against them. This tests the
judging, not the writing of a rubric. The human side is the track's primary score, vital strict.

`--n-systems` takes the best and the worst system by the human score and one at random from each rank band
between them (10 by default), and `--n-topics` a difficulty-stratified sample of topics (all 56 by default).
The output is each system's human score and Elo rating, Kendall's tau and Spearman between the two, and how
often a game's winner is the answer with the higher human score.

## Datasets

| `--dataset` | Pairs | Queries | Labels | Source |
|---|---|---|---|---|
| `llmjudge` (default) | 7,263 dev, 4,423 test | 25 per split | 0-3, NIST | [LLMJudge](https://github.com/llm4eval/LLMJudge-benchmark), TREC DL 2023 passages |
| `trec_rag24` | 20,277 | 86 | 0-3, NIST | [TREC 2024 RAG](https://trec.nist.gov/data/rag2024.html) retrieval qrels, MS MARCO v2.1 segments |

Neither needs a TREC password. `trec_rag24` reads its passages from the 27 GB segment corpus without storing
it: a segment id ends in the byte offset of its line, so the loader streams each gzipped file and keeps the
20,193 judged segments, 28 MB. It resumes per corpus file. The qrels carry one placeholder id,
`msmarco_v2.1_doc_01_23#45_67`, whose 6 judgments are dropped.

Both the dev and the test split were used while tuning the prompts below, so neither is held out.

## Defaults the benchmark settled

Numbers are Spearman correlations between the rounded labels and the human labels, on 500 pairs from 25
queries per split, with 95% intervals that resample queries.

| Default | Compared with | dev | test |
|---|---|---|---|
| `reasoner` grades by what a document contributes to an answer | grading by topic, which put half of the pairs at "somewhat relevant" where the assessors put 19% to 28% | +0.05, interval includes zero | +0.04, interval includes zero |
| `jev` asks one yes/no question (`boolean_question=True`) | a score question over the relevance grades | +0.11 (0.05 to 0.18) | +0.05 (-0.01 to 0.12) |
| `jev` asks RDNAM's report test | "does the document help answer the question" | +0.05 (0.01 to 0.09) | +0.04 (-0.03 to 0.11) |

## Asking Jev about several documents in one request

The Jev providers ask prompts that share a batch key in one request (`batch_size`, default 10). The `jev`
retrieval evaluators key their prompts by query. Measured on 2026-09-20 with the `typesafe` provider and the
`jev` evaluator on the 500-pair LLMJudge samples, `n_processes=16`:

| `batch_size` | Input tokens, dev | Spearman dev (95% interval) | Input tokens, test | Spearman test (95% interval) |
|---|---|---|---|---|
| 1 | 252,761 | 0.502 (0.418 to 0.586) | 235,522 | 0.486 (0.373 to 0.598) |
| 10 | 158,757 | 0.577 (0.479 to 0.666) | 138,722 | 0.531 (0.438 to 0.609) |

Batching cuts the input tokens by about 40% and does not change the wall time at the same `n_processes`,
because fewer requests are in flight. The agreement only improves when the batch holds documents of one query.
Asked outside the library, 10 documents of mixed queries per request scored 0.511 on dev and 0.452 on test,
against 0.550 and 0.570 for the same merge in query order, which is why a prompt without a batch key is never
merged. Both splits were tuned on, and the intervals overlap.

## Held-out check on TREC 2024 RAG

No prompt was tuned on this dataset. 2,001 label-stratified pairs over all 86 queries, judged on 2026-09-20 at
16 parallel calls. Prices are the Vercel AI Gateway's, in USD per million input, output and cached tokens:
0.20, 1.20 and 0.02 for `gpt-5.6-luna`, 0.042, 0 and 0.042 for Jev.

| Evaluator | Model | Spearman (95% interval) | Raw-score Spearman | Kappa, binary | Alpha | USD / 1k pairs | Evaluations / s |
|---|---|---|---|---|---|---|---|
| `rdnam` | gpt-5.6-luna | 0.525 (0.457 to 0.595) | 0.525 | 0.411 | 0.497 | 0.269 | 7.8 |
| `reasoner` | gpt-5.6-luna | 0.518 (0.446 to 0.587) | 0.518 | 0.416 | 0.483 | 0.234 | 8.3 |
| `jev` | typesafe-ai/jev | 0.507 (0.436 to 0.578) | 0.561 | 0.397 | 0.404 | 0.025 | 8.0 to 10.6 |
| `jev_rdnam` | typesafe-ai/jev | 0.506 (0.427 to 0.581) | 0.578 | 0.357 | 0.487 | 0.034 | 19.4 to 44.1 |

Every paired difference between two rows has an interval that includes zero on the rounded labels. The
raw-score column favours Jev because a fractional score breaks ties that a three-level label cannot. All four
judges are more generous than the assessors, who gave grade 0 to 37% of the pairs where the judges gave it to
10% to 22%.
