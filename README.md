
<h1 align="center">
<img style="vertical-align:middle" src="https://raw.githubusercontent.com/zetaalphavector/RAGElo/master/docs/images/RAGElo_logo.png" height="200">
</h1>

<p  align="center" >
<i> Elo-based RAG Agent evaluator </i>
</p>


**RAGElo**[^1] is a streamlined toolkit for evaluating Retrieval Augmented Generation (RAG)-powered Large Language Models (LLMs) question answering agents using the Elo rating system.

While it has become easier to prototype and incorporate generative LLMs in production, evaluation is still the most challenging part of the solution. Comparing different outputs from multiple prompt and pipeline variations to a "gold standard" is not easy. Still, we can ask a powerful LLM to judge between pairs of answers and a set of questions.

This led us to develop a simple tool for tournament-style Elo ranking of LLM outputs. By comparing answers from different RAG pipelines and prompts over multiple questions, RAGElo computes a ranking of the different settings, providing a good overview of what works (and what doesn't).


## ⚙️ Installation
For using RAGElo as a Python library or as CLI, install it using pip:

```bash
pip install ragelo
```

When working from source we recommend an isolated environment (e.g., `uv venv && uv pip install -e '.[dev]'`). The project's Python lives at `.venv/bin/python`.

Environment variables and providers:
- OpenAI requires `OPENAI_API_KEY`. Set it in your shell or load it via dotenv before invoking the CLI.
- Ollama is supported for local models (`--llm-provider-name ollama`).
- The **Instructor provider** enables multi-provider support (Anthropic, Mistral, Cohere, and more) via the [`instructor`](https://github.com/jxnl/instructor) library. Install the extra and the relevant SDK:
  ```bash
  pip install 'ragelo[instructor]' anthropic   # for Anthropic/Claude
  pip install 'ragelo[instructor]' mistralai   # for Mistral
  pip install 'ragelo[instructor]' cohere      # for Cohere
  ```
  Then set the matching API key environment variable (e.g. `ANTHROPIC_API_KEY`).

## 🚀 Library Quickstart

To use RAGElo as a library, all you need to do is import RAGElo, initialize an `Evaluator` and call either `evaluate()` for evaluating a retrieved document or an LLM answer, or `evaluate_experiment()` to evaluate a full experiment. For example, using the `RDNAM` retrieval evaluator from the [Thomas et al. (2023)](https://arxiv.org/abs/2309.10621) paper on using GPT-4 for annotating retrieval results:

```python
from ragelo import get_retrieval_evaluator

evaluator = get_retrieval_evaluator("RDNAM", llm_provider="openai")
result = evaluator.evaluate(query="What is the capital of France?", document='Lyon is the second largest city in France.')
print(result.answer)
# Output: RDNAMEvaluationAnswer(reasoning='...', score=1.0, intent_match=None, trustworthiness=None)
print(result.answer.score)
# Output: 1.0
print(result.answer.model_dump_json())
# Output: '{"reasoning": "...", "score": 1.0, "intent_match": null, "trustworthiness": null}'
```

In most cases `result.answer` contains a `BaseModel` from Pydantic with the parsed judge response. For more details, check the [answer_formats.py](https://github.com/zetaalphavector/RAGElo/blob/master/ragelo/types/answer_formats.py) file. 

### 🔄 Evaluating a single query incrementally

If queries arrive one at a time (e.g., in an online or streaming workflow), you can evaluate all evaluables for a single query without constructing a full experiment:

```python
from ragelo import get_retrieval_evaluator, get_answer_evaluator
from ragelo.types.query import Query
from ragelo.types.evaluables import Document, AgentAnswer

query = Query(qid="q0", query="What is the capital of Brazil?")
query.add_retrieved_doc(Document(qid="q0", did="d0", text="Brasília is the capital of Brazil."))
query.add_agent_answer(AgentAnswer(qid="q0", agent="agent1", text="Brasília."))

retrieval_evaluator = get_retrieval_evaluator("reasoner", llm_provider="openai")
retrieval_evaluator.evaluate_all_evaluables(query)

# Each document now has an evaluation attached
for doc in query.retrieved_docs.values():
    print(doc.did, doc.evaluations)
```

This calls the same evaluation logic as `evaluate_experiment` but scoped to one query, making it suitable for incremental pipelines.

### 📜 Evaluating multiple documents or answers

RAGElo supports `Experiments` to keep track of which documents and answers were already evaluated and to compute overall scores for each Agent:

```python
from ragelo import Experiment, get_retrieval_evaluator, get_answer_evaluator, get_agent_ranker, get_llm_provider

experiment = Experiment(experiment_name="A_really_cool_RAGElo_experiment")
# Add two user queries. Alternatively, we can load them from a csv file with .add_queries_from_csv()
experiment.add_query("What is the capital of Brazil?", query_id="q0")
experiment.add_query("What is the capital of France?", query_id="q1")

# Add four documents retrieved for these queries. Alternatively, we can load them from a csv file with .add_documents_from_csv()
experiment.add_retrieved_doc("Brasília is the capital of Brazil", query_id="q0", doc_id="d0")
experiment.add_retrieved_doc("Rio de Janeiro used to be the capital of Brazil.", query_id="q0", doc_id="d1")
experiment.add_retrieved_doc("Paris is the capital of France.", query_id="q1", doc_id="d2")
experiment.add_retrieved_doc("Lyon is the second largest city in France.", query_id="q1", doc_id="d3")

# Add the answers generated by agents
experiment.add_agent_answer("Brasília is the capital of Brazil, according to [0].", agent="agent1", query_id="q0")
experiment.add_agent_answer("According to [1], Rio de Janeiro used to be the capital of Brazil, until the 60s.", agent="agent2", query_id="q0")
experiment.add_agent_answer("Paris is the capital of France, according to [2].", agent="agent1", query_id="q1")
experiment.add_agent_answer("According to [3], Lyon is the second largest city in France. Meanwhile, Paris is its capital [2].", agent="agent2", query_id="q1")

llm_provider = get_llm_provider("openai", model="gpt-4.1-nano")

# Or use the Instructor provider to run evaluations against Anthropic Claude (requires pip install 'ragelo[instructor]' anthropic):
# llm_provider = get_llm_provider("instructor", model="anthropic/claude-sonnet-4-20250514")

retrieval_evaluator = get_retrieval_evaluator("reasoner", llm_provider, rich_print=True)
answer_evaluator = get_answer_evaluator("pairwise", llm_provider, rich_print=True)

elo_ranker = get_agent_ranker("elo", show_results=True)

# Evaluate the retrieval results.
retrieval_evaluator.evaluate_experiment(experiment)

# With the retrieved documents evaluated, evaluate the quality of the answers. using the pairwise evaluator
answer_evaluator.evaluate_experiment(experiment)

# Run the ELO ranker to score the agents
elo_ranker.run(experiment)
# Output:
    ------- Agents Elo Ratings -------
    agent1         : 1035.7(±2.9)
    agent2         : 961.3(±2.9)
```

The experiment is save as a JSON in `ragelo_cache/experiment_name.json`. 

### 🛠️ Using a custom prompt and injecting metadata
For a more complete example, we can evaluate with a custom prompt, and inject metadata into our evaluation prompt:

```python
from pydantic import BaseModel, Field
from ragelo import get_retrieval_evaluator

system_prompt = """You are a helpful assistant for evaluating the relevance of a retrieved document to a user query.
You should pay extra attention to how **recent** a document is. A document older than 5 years is considered outdated.

The answer should be evaluated according to its recency, truthfulness, and relevance to the user query.
"""

user_prompt = """
User query: {{ query.query }}

Retrieved document: {{ document.text }}

The document has a date of {{ document.metadata.date }}.
Today is {{ query.metadata.today_date }}.
"""
class ResponseSchema(BaseModel):
    relevance: int = Field(description="An integer, either 0 or 1. 0 if the document is irrelevant, 1 if it is relevant.")
    recency: int = Field(description="An integer, either 0 or 1. 0 if the document is outdated, 1 if it is recent.")
    truthfulness: int = Field(description="An integer, either 0 or 1. 0 if the document is false, 1 if it is true.")
    reasoning: str = Field(description="A short explanation of why you think the document is relevant or irrelevant.")



evaluator = get_retrieval_evaluator(
    "custom_prompt", # name of the retrieval evaluator
    llm_provider="openai", # Which LLM provider to use
    system_prompt=system_prompt, # your custom prompt
    user_prompt=user_prompt, # your custom prompt
    result_type=ResponseSchema, # The response schema for the LLM. 
)

result = evaluator.evaluate(
    query="What is the capital of Brazil?", # The user query
    document="Rio de Janeiro is the capital of Brazil.", # The retrieved document
    query_metadata={"today_date": "08-04-2024"}, # Some metadata for the query
    doc_metadata={"date": "04-03-1950"}, # Some metadata for the document
)

result.answer.model_dump_json(indent=2)
# Output: 
    '{
        "relevance": 0,
        "recency": 0,
        "truthfulness": 0,
        "reasoning": "The document is outdated and incorrect. Rio de Janeiro was the capital of Brazil until 1960 when it was changed to Brasília."
    }'
```
Note that, in this example, we passed to the `evaluate` method two dictionaries with metadata for the query and the document. This metadata is injected into the prompt by matching their keys into the placeholders in the prompt (note the `document.metadata.date` and `query.metadata.today_date` templates.)

For a comprehensive example of how to use RAGElo, see the [docs/examples/notebooks/rag_eval.ipynb](https://github.com/zetaalphavector/RAGElo/blob/master/docs/examples/notebooks/rag_eval.ipynb) notebook.

## 🚀 CLI Quickstart
After installing RAGElo as a CLI app (and exporting the appropriate LLM provider credentials, e.g., `OPENAI_API_KEY`), you can run it with the following command:
```bash
ragelo run-all \
  queries.csv documents.csv answers.csv \
  --data-dir tests/data/ \
  --experiment-name tutorial \
  --output-file tutorial.json \
  --show-results
```

With `--show-results` enabled you will see outputs such as:

```
Loaded 2 queries from .../tests/data/queries.csv
Loaded 4 new documents from .../tests/data/documents.csv
Loaded 4 answers from .../tests/data/answers.csv
Evaluating Retrieved documents ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4/4
✅ Done!
🔎 Query ID: 0
📜 Document ID: 0
Parsed Answer: Very relevant: The document directly answers the user question by stating that Brasília is the capital of Brazil.

🔎 Query ID: 0
📜 Document ID: 1
Parsed Answer: Somewhat relevant: The document mentions a former capital of Brazil but does not provide the current capital.

🔎 Query ID: 1
📜 Document ID: 2
Parsed Answer: Very relevant: The document clearly states that Paris is the capital of France, directly answering the user question.

🔎 Query ID: 1
📜 Document ID: 3
Parsed Answer: Not relevant: The document does not provide information about the capital of France.

Evaluating Retrieved documents 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4/4  [ 0:00:02 < 0:00:00 , 2 it/s ]
✅ Done!
Total evaluations: 4
🔎 Query ID: 0
 agent1              🆚   agent2
Parsed Answer: A

🔎 Query ID: 1
 agent1              🆚   agent2
Parsed Answer: A

Evaluating Agent Answers 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2/2  [ 0:00:09 < 0:00:00 , 0 it/s ]
✅ Done!
Total evaluations: 2
------- Agents Elo Ratings -------
agent1         : 1033.0(±0.0)
agent2         : 966.0(±0.0)
```

By default, evaluations are persisted to `ragelo_cache/<experiment>.json` alongside incremental results in `ragelo_cache/<experiment>_results.jsonl`. Passing `--output-file` writes the experiment JSON (without evaluator traces) to a custom location.

In this example, the output file is a JSON file with the experiment definition and tournament summary. It can be loaded directly as a new `Experiment` object:

```python
experiment = Experiment.load("experiment", "experiment.json")
```

When running as a CLI, RAGElo expects the input files as CSV files. Specifically, it expects a csv file with the user queries, one with the documents retrieved by the retrieval system and one of the answers each agent produced. These files can be passed with the parameters `--queries_csv_file`, `--documents_csv_file` and `--answers_csv_file`, respectively, or directly as positional arguments.

CSV columns and inference:
- Queries: `qid`, `query` (infers `qid` if missing)
- Documents: `qid`, `did`, `document` (infers `qid`/`did` if missing)
- Answers: `qid`, `agent`, `answer`
Extra columns are captured as metadata and available to prompts.

Here are some examples of their expected formats:
`queries.csv`:
```csv
qid,query
0, What is the capital of Brazil?
1, What is the capital of France?

```

`documents.csv`:
```csv
qid,did,document
0,0, Brasília is the capital of Brazil.
0,1, Rio de Janeiro used to be the capital of Brazil.
1,2, Paris is the capital of France.
1,3, Lyon is the second largest city in France.
```

`answers.csv`:
```csv
qid,agent,answer
0, agent1,"Brasília is the capital of Brazil, according to [0]."
0, agent2,"According to [1], Rio de Janeiro used to be the capital of Brazil, until the 60s."
1, agent1,"Paris is the capital of France, according to [2]."
1, agent2,"According to [3], Lyon is the second largest city in France. Meanwhile, Paris is its capital [2]."
```


## 🧩 Components
While **RAGElo** can be used end-to-end (`run-all`), you can also drive individual CLI components.

### 📜 `retrieval-evaluator`
The `retrieval-evaluator` tool annotates retrieved documents based on their relevance to the user query. This is done regardless of the answers provided by any Agent. As an example, for calling the `Reasoner` retrieval evaluator (reasoner only outputs the reasoning why a document is relevant or not) we can use:

```bash
ragelo retrieval-evaluator reasoner \
  queries.csv documents.csv \
  --data-dir tests/data/ \
  --experiment-name experiment \
  --output-file experiment-docs.json \
  --show-results
```
Each run updates the experiment cache and appends evaluation traces to `<experiment>_results.jsonl`. If all documents already have evaluations you will see an informational message unless `--force` is provided.

Domain expert example:
```bash
ragelo retrieval-evaluator domain-expert \
  queries.csv documents.csv \
  --data-dir tests/data/ \
  --experiment-name experiment \
  --expert-in "Chemical Engineering" \
  --company "ChemCorp" \
  --output-file experiment-docs.json \
  --show-results
```

RDNAM example:
```bash
ragelo retrieval-evaluator rdnam \
  queries.csv documents.csv \
  --data-dir tests/data/ \
  --experiment-name experiment \
  --output-file experiment-docs.json \
  --show-results
```

### 💬 `answer-evaluator`

The `answer-evaluator` subcommands annotate agent answers. The default `pairwise` mode compares answers two at a time and can optionally inject reasoning annotations:

```bash
ragelo answer-evaluator pairwise \
  queries.csv documents.csv answers.csv \
  --data-dir tests/data/ \
  --experiment-name experiment \
  --output-file experiment-answers.json \
  --add-reasoning \
  --show-results
```

If `--add-reasoning` is supplied the CLI will run the `reasoner` retrieval evaluator first, include the relevance scores in the prompts, and then proceed with pairwise games. Newly created games are tracked inside the experiment and re-used by the Elo ranker.

Domain expert pairwise example:
```bash
ragelo answer-evaluator expert-pairwise \
  queries.csv documents.csv answers.csv \
  --data-dir tests/data/ \
  --experiment-name experiment \
  --expert-in "Healthcare" \
  --add-reasoning \
  --output-file experiment-answers.json \
  --show-results
```

Concurrency and Rich output:
- Use `--n-processes` to control parallel LLM calls.
- Use `--no-rich-print` in CI to avoid live display issues.

Reproducibility tips:
- Pairwise sampling (`n_games_per_query`) is randomized; persist experiment JSON/JSONL to stabilize comparisons.

Evaluating retrieval metrics (optional):
```python
from ragelo import Experiment, NullStorageBackend
exp = Experiment(experiment_name="my_exp", storage_backend=NullStorageBackend())
# load queries/docs/answers and evaluations...
exp.evaluate_retrieval(metrics=["Precision@10", "nDCG@10"], relevance_threshold=1)
```

## 🙋 Contributing

To install the development dependencies, download the repository and run the following:

```bash
git clone https://github.com/zeta-alpha/ragelo && cd ragelo
uv pip install -e '.[dev]'
```

This will install the requirement dependencies in an editable mode (i.e., any changes to the code don't need to be rebuilt.)
For building a new version, use the `build` command:

```bash
python -m build
```

### ✅ TODO
- [ ] Add full documentation of all implemented Evaluators
- [X] Add CI/CD for publishing
- [x] Add option to few-shot examples (Undocumented, yet)
- [x] Testing!
- [x] Publish on PyPi
- [x] Add more document evaluators
- [x] Split Elo evaluator
- [x] Install as standalone CLI

[^1]: The RAGElo logo was created using Dall-E 3 and GPT-4 with the following prompt: "Vector logo design for a toolkit named 'RAGElo'. The logo should have bold, modern typography with emphasis on 'RAG' in a contrasting color. Include a minimalist icon symbolizing retrieval or ranking."
