# Benchmarks

`run_llmjudge.py` and `run_llmjudge_pairwise.py` judge passages and compare the judgments with the human
labels. Each table prints how many pairs failed, and the agreement is computed over the pairs every row judged.

```sh
uv run --python 3.13 python -m benchmarks.run_llmjudge --model gpt-5.6-luna --n-pairs 500
uv run --python 3.13 python -m benchmarks.run_llmjudge --dataset trec_rag24 --model gpt-5.6-luna --n-pairs 2000
uv run --python 3.13 python -m benchmarks.run_llmjudge_pairwise --model typesafe-ai/jev
```

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
