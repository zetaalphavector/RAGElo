
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

## 🚀 Library Quickstart

To use RAGElo as a library, all you need to do is import RAGElo, initialize an `Evaluator` and call either `evaluate()` for evaluating a retrieved document or an LLM answer, or `batch_evaluate()` to evaluate multiple responses at once. For example, using the `RDNAM` retrieval evaluator from the [Thomas et al. (2023)](https://arxiv.org/abs/2309.10621) paper on using GPT-4 for annotating retrieval results:

```python
from ragelo import get_retrieval_evaluator

evaluator = get_retrieval_evaluator("RDNAM", llm_provider="openai")
result = evaluator.evaluate(query="What is the capital of France?", document='Lyon is the second largest city in France.')
print(result.answer)
# Output: {'overall': 1}
print(result.answer["overall"])
# Output: 1
print(result.raw_answer)
# Output: '{"overall": 1"}'
```

For a more complete example, we can evaluate with a custom prompt, and inject metadata into our evaluation prompt:

```python
from ragelo import get_retrieval_evaluator

prompt = """You are a helpful assistant for evaluating the relevance of a retrieved document to a user query.
You should pay extra attention to how **recent** a document is. A document older than 5 years is considered outdated.

The answer should be evaluated according to its recency, truthfulness, and relevance to the user query.

User query: {q}

Retrieved document: {d}

The document has a date of {document_date}.
Today is {today_date}.
"""
response_schema = {
    "relevance": "An integer, either 0 or 1. 0 if the document is irrelevant, 1 if it is relevant.",
    "recency": "An integer, either 0 or 1. 0 if the document is outdated, 1 if it is recent.",
    "truthfulness": "An integer, either 0 or 1. 0 if the document is false, 1 if it is true.",
    "reasoning": "A short explanation of why you think the document is relevant or irrelevant.",
}
response_format = "json"

evaluator = get_retrieval_evaluator(
    "custom_prompt", # name of the retrieval evaluator
    llm_provider="openai", # Which LLM provider to use
    prompt=prompt, # your custom prompt
    query_placeholder="q", # the placeholder for the query in the prompt
    document_placeholder="d", # the placeholder for the document in the prompt
    llm_answer_format=response_format, # The format of the answer. Can be either `text`, if you expect plain text to be returned, `JSON` if the answer should be in JSON format, or `structured`, if you provide a Pydantic BaseModel as the response_schema.
    llm_response_schema=response_schema, # The response schema for the LLM. Required if the llm_answer_format is structured and recommended for JSON.
    seed=42, # The seed for the LLM. Used to ensure we get the same answer for the same query and document pair.
)

result = evaluator.evaluate(
    query="What is the capital of Brazil?", # The user query
    document="Rio de Janeiro is the capital of Brazil.", # The retrieved document
    query_metadata={"today_date": "08-04-2024"}, # Some metadata for the query
    doc_metadata={"document_date": "04-03-1950"}, # Some metadata for the document
)

answer
{'relevance': 0,
 'recency': 0,
 'truthfulness': 0,
 'reasoning': 'The document is outdated and incorrect. Rio de Janeiro was the capital of Brazil until 1960 when it was changed to Brasília.'}
```
Note that, in this example, we passed to the `evaluate` method two dictionaries with metadata for the query and the document. This metadata is injected into the prompt by matching their keys into the placeholders in the prompt.

For a comprehensive example of how to use RAGElo, see the [docs/examples/notebooks/rag_eval.ipynb](https://github.com/zetaalphavector/RAGElo/blob/master/docs/examples/notebooks/rag_eval.ipynb) notebook.

## 🚀 CLI Quickstart
After installing RAGElo as a CLI app, you can run it with the following command:
```bash
ragelo run-all queries.csv documents.csv answers.csv --data-dir tests/data/ --experiment-name experiment --output-file experiment.json

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

In this example, the output file is a JSON file with all the annotations performed by the evaluators and the final Elo ratings. These can also be loaded directly as a new `Experiment` object:

```python
experiment = Experiment.load("experiment", "experiment.json")
```

When running as a CLI, RAGElo expects the input files as CSV files. Specifically, it expects a csv file with the user queries, one with the documents retrieved by the retrieval system and one of the answers each agent produced. These files can be passed with the parameters `--queries_csv_file`, `--documents_csv_file` and `--answers_csv_file`, respectively, or directly as positional arguments.

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
While **RAGElo** can be used as either an end-to-end tool or by calling individual CLI components.

### 📜 `retrieval-evaluator`
The `retrieval-evaluator` tool annotates retrieved documents based on their relevance to the user query. This is done regardless of the answers provided by any Agent. As an example, for calling the `Reasoner` retrieval evaluator (reasoner only outputs the reasoning why a document is relevant or not) we can use:

```bash
ragelo retrieval-evaluator reasoner queries.csv documents.csv --data-dir tests/data/ --experiment-name experiment
```
The output file changes according to the evaluator used. In general it will have one row per document evaluator, with the query_id, document_id, the raw LLM answer and the parsed answer. An example of the output for the reasoner is found here: [tests/data/reasonings.csv](https://github.com/zetaalphavector/RAGElo/blob/master/tests/data/reasonings.csv).

### 💬 `answers-annotator`

The `answers-annotator` tool annotates the answers generated by the Agents, taking the quality of the documents retrieved by the retrieval pipeline. By default, it uses the `Pairwise` annotator, which generates `k` random pairs of answers for each query and chooses the best answer based on the relevant documents cited in the answer. If the experiment already exists with annotations for the documents, it will try to load these and inject into the prompts for a better context for the LLM. Otherwise, you can pass the `--add-reasoning` flag to run the `Reasoner` retrieval evaluator first.

```bash
ragelo answer-evaluator pairwise queries.csv documents.csv answers.csv --data-dir tests/data/ --experiment-name experiment --add-reasoning
```

## 🙋 Contributing

To install the development dependencies, download the repository and run the following:

```bash
git clone https://github.com/zeta-alpha/ragelo && cd ragelo
pip install -e '.[dev]'
```

This will install the requirement dependencies in an editable mode (i.e., any changes to the code don't need to be rebuilt.)
For building a new version, use the `build` command:

```bash
python -m build
```

### ✅ TODO
- [ ] Add CI/CD for publishing
- [ ] Add full documentation of all implemented Evaluators
- [x] Add option to few-shot examples (Undocumented, yet)
- [x] Testing!
- [x] Publish on PyPi
- [x] Add more document evaluators
- [x] Split Elo evaluator
- [x] Install as standalone CLI

[^1]: The RAGElo logo was created using Dall-E 3 and GPT-4 with the following prompt: "Vector logo design for a toolkit named 'RAGElo'. The logo should have bold, modern typography with emphasis on 'RAG' in a contrasting color. Include a minimalist icon symbolizing retrieval or ranking."
