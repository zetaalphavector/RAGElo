# Benchmarks

`run_llmjudge.py` and `run_llmjudge_pairwise.py` judge the [LLMJudge](https://github.com/llm4eval/LLMJudge-benchmark)
passages and compare the judgments with the human labels. Each table prints how many pairs failed, and the
agreement is computed over the pairs every row judged.

```sh
uv run --python 3.13 python -m benchmarks.run_llmjudge --model gpt-5.6-luna --n-pairs 500
uv run --python 3.13 python -m benchmarks.run_llmjudge_pairwise --model typesafe-ai/jev
```

Both the dev and the test split were used while tuning the prompts below, so neither is held out.

## Defaults the benchmark settled

Numbers are Spearman correlations between the rounded labels and the human labels, on 500 pairs from 25
queries per split, with 95% intervals that resample queries.

| Default | Compared with | dev | test |
|---|---|---|---|
| `reasoner` grades by what a document contributes to an answer | grading by topic, which put half of the pairs at "somewhat relevant" where the assessors put 19% to 28% | +0.05, interval includes zero | +0.04, interval includes zero |
| `jev` asks one yes/no question (`boolean_question=True`) | a score question over the relevance grades | +0.11 (0.05 to 0.18) | +0.05 (-0.01 to 0.12) |
| `jev` asks RDNAM's report test | "does the document help answer the question" | +0.05 (0.01 to 0.09) | +0.04 (-0.03 to 0.11) |
