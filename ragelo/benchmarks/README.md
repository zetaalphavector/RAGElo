# Benchmarks

`ragelo benchmark` judges a public dataset with RAGElo's evaluators and reports how the judgments agree with
the human assessors, with tokens, cost and throughput per evaluator and model.

## Usage

Install SciPy with the dev group or `pip install 'ragelo[benchmarks]'`, and set the key of the provider:
`OPENAI_API_KEY` for `openai`, `AI_GATEWAY_API_KEY` for `vercel` and `vercel-jev`, `TYPESAFE_API_KEY` for
`typesafe`. Data is public and downloaded on first use.

```sh
ragelo benchmark llmjudge --n-samples 500 --models gpt-5.6-luna \
    --evaluators reasoner --evaluators rdnam --prices gpt-5.6-luna=0.20,1.20,0.02

ragelo benchmark trec-rag24 --n-samples 2000 --llm-provider-name vercel-jev \
    --models typesafe-ai/jev --evaluators jev_boolean --evaluators jev_rdnam

ragelo benchmark llmjudge-pairwise --llm-provider-name vercel-jev \
    --models typesafe-ai/jev --evaluators jev_pairwise

ragelo benchmark trec-rag24-answers --llm-provider-name vercel-jev \
    --models typesafe-ai/jev --evaluators jev_pairwise --evaluators jev_rubric_pairwise
```

| Option | Meaning |
|---|---|
| `--models` | The models to judge with. Repeat for several. |
| `--evaluators` | The evaluators to judge with. Repeat for several. Every model and evaluator pair is one row of the report. |
| `--llm-provider-name` | `openai` by default. |
| `--split` | `dev` (default) or `test`, on the LLMJudge datasets. |
| `--n-samples` | Judge a sample instead of everything. The unit is in the datasets table. |
| `--seed` | Seed of the sample and of the Elo tournaments, 42 by default. |
| `--n-systems` | `trec-rag24-answers` only, 10 by default. |
| `--prices` | `MODEL=INPUT,OUTPUT,CACHED` in USD per million tokens. Without it the report has tokens only. |
| `--n-processes` | Parallel evaluations, 16 by default. |
| `--force` | Judge again over the cache. |
| `--tag` | Suffix of the experiment names, to judge again beside an old cache. |
| `--data-dir` | `benchmarks/data/<dataset>` by default. |
| `--output-dir` | Where the judgments are cached, `benchmarks/results/<dataset>` by default. |

Judgments are cached in one experiment per sample, evaluator, model and tag. A re-run judges only what is
missing or failed, and reports the throughput of the runs that judged at least 100 items.

## Datasets

| Command | Evaluators | Judges | `--n-samples` draws | Compared with |
|---|---|---|---|---|
| `llmjudge` | retrieval | 7,263 dev and 4,423 test query-passage pairs, 25 queries per split | pairs, label-stratified | NIST 0-3 labels on TREC DL 2023 passages, from [LLMJudge](https://github.com/llm4eval/LLMJudge-benchmark) |
| `trec-rag24` | retrieval | 20,277 query-segment pairs, 86 queries | pairs, label-stratified | NIST 0-3 labels of the [TREC 2024 RAG](https://trec.nist.gov/data/rag2024.html) retrieval task |
| `llmjudge-pairwise` | pairwise answer | games between two passages of one LLMJudge query | games, 500 by default | which passage NIST graded higher |
| `trec-rag24-answers` | pairwise answer | an Elo tournament between `--n-systems` of the 45 systems, on their 2,471 answers to 56 topics | topics, difficulty-stratified | the systems' ranking by the track's vital strict nugget score |

A retrieval dataset takes the evaluator names in [`variants.py`](variants.py), and reports Cohen's kappa,
Krippendorff's alpha and Spearman against the labels, ranked by Spearman with a 95% interval that resamples
queries. A pairwise dataset takes the name of any pairwise answer evaluator. On `trec-rag24-answers` the
evaluators with "rubric" in their name get each topic's nuggets as its rubric, weighted 1 for vital and 0.5
for okay.

The first `trec-rag24` run streams the 27 GB MS MARCO v2.1 segment corpus and stores the judged segments,
28 MB. It resumes per corpus file.

## Adding to the suite

A retrieval configuration is an entry in `RETRIEVAL_VARIANTS` in [`variants.py`](variants.py). A dataset is a
subclass of `Dataset`, `RetrievalDataset` or `PairwiseDataset` in [`datasets/`](datasets), registered with
`@DatasetFactory.register` under a `BenchmarkDatasetTypes` name, imported in `datasets/__init__.py` and given a
command in `ragelo/cli/benchmark_cli.py`.

## Results

Judged on 2026-09-20 at 16 parallel evaluations: `gpt-5.6-luna` through `openai`, Jev through `vercel-jev`
with its default batching. Prices are the Vercel AI Gateway's on that day, in USD per million input, output
and cached tokens: 0.20, 1.20 and 0.02 for `gpt-5.6-luna`, 0.042, 0 and 0.042 for Jev.

### Retrieval: `trec-rag24`, 2,001 pairs over 86 queries

`--n-samples 2000`, seed 42.

| Evaluator | Model | Spearman (95% interval) | Kappa, binary | Alpha | USD / 1k pairs | Evaluations / s |
|---|---|---|---|---|---|---|
| `rdnam` | gpt-5.6-luna | 0.525 (0.457 to 0.595) | 0.411 | 0.497 | 0.269 | 7.8 |
| `reasoner` | gpt-5.6-luna | 0.518 (0.446 to 0.587) | 0.416 | 0.483 | 0.234 | 8.3 |
| `jev_boolean` | typesafe-ai/jev | 0.502 (0.432 to 0.568) | 0.396 | 0.459 | 0.018 | 15.3 |
| `jev_rdnam` | typesafe-ai/jev | 0.487 (0.415 to 0.553) | 0.337 | 0.469 | 0.026 | 38.2 |

### Answers: `trec-rag24-answers`, 10 systems, 56 topics, 2,520 games

Seed 42. Kendall's tau is between the Elo ratings and the human scores of the systems. A game agrees when its
winner is the answer with the higher human score.

| Evaluator | Model | Kendall's tau | Games agreeing | USD / 1k games | Games / s |
|---|---|---|---|---|---|
| `rubric_pairwise` | gpt-5.6-luna | 0.956 | 89.0% | 5.41 | 0.46 |
| `jev_rubric_pairwise` | typesafe-ai/jev | 0.911 | 87.1% | 0.200 | 8.0 |
| `jev_pairwise` | typesafe-ai/jev | 0.911 | 77.9% | 0.067 | 8.1 |
| `pairwise` | gpt-5.6-luna | 0.778 | 72.9% | 2.21 | 0.98 |
