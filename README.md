
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

```bash
pip install ragelo
```

If you want to use RAGElo as a standalone CLI app, use the `[cli]` tag:

```bash
pip install ragelo[cli]
```

## 🚀 Quickstart 
After installing RAGElo as a CLI app, you can run it with the following command:
```bash
ragelo run-all queries.csv documents.csv answers.csv

---------- Agent Scores by Elo ranking ----------
 agent1        : 1026.7
 agent2        : 973.3
```

We need three files for running an end-to-end evaluation: `queries.csv`, `documents.csv`, and `answers.csv`:

`queries.csv`: 
```csv
query_id,query
0, "What is the capital of Brazil?"
1, "What is the capital of France?"
```

`documents.csv`:
```csv
query_id,doc_id,document_text
0,0, "Brasília is the capital of Brazil."
0,1, "Rio de Janeiro used to be the capital of Brazil."
1,2, "Paris is the capital of France."
1,3, "Lyon is the second largest city in France."
```

`answers.csv`:
```csv
query_id,agent,answer
0, agent1, "Brasília is the capital of Brazil, according to [0]."
0, agent2, "According to [1], Rio de Janeiro used to be the capital of Brazil until 1960."
1, agent1, "Paris is the capital of France, according to [2]."
1, agent2, "Lyon is the second largest city in France, according to [3]."
```

The OpenAI API key should be set as an environment variable (`OPENAI_API_KEY`). Alternatively, you can set a credentials file and pass it as an option to `ragelo`:

`credentials.txt`:
```
OPENAI_API_KEY=<your_key_here>
```

```bash
ragelo --credentials credentials.txt run-all queries.csv documents.csv answers.csv 
```

## 🧩 Components
While **RAGElo** is meant to be used as an end-to-end tool, we can also invoke each of its components individually:

### 📜 `retrieval-annotator`
The `retrieval-annotator` tool annotates retrieved documents based on their relevance to the user query. This is done regardless of the answers provided by the Agents. By default, it uses the `reasoner` annotator, which only outputs the reasoning for the relevance judgment:

```bash
ragelo retrieval-annotator queries.csv documents.csv reasonings.csv
```
The `reasonings.csv` output file is a csv file with query_id, document_id and answer columns: [tests/data/reasonings.csv](https://github.com/zetaalphavector/RAGElo/blob/master/tests/data/reasonings.csv).

### 💬 `answers-annotator`

The `answers-annotator` tool annotates the answers generated by the Agents, taking the quality of the documents retrieved by the retrieval pipeline. By default, it uses the `PairwiseWithReasoning` annotator, which generates `k` random pairs of answers for each query and chooses the best answer based on the relevant documents cited in the answer. It relies on the `reasonings.csv` file generated by the `documents-annotator`:

```bash
ragelo answers-annotator queries.csv answers.csv reasonings.csv answers_eval.jsonl
```

The `answers_eval.jsonl` output file is a JSONL file with each line containing the prompt for evaluating the pair of answers, the output of the annotator, and the best answer. An output file example is provided at [tests/data/answers_eval.jsonl](https://github.com/zetaalphavector/RAGElo/blob/master/tests/data/answers_eval.jsonl)
 
### 🏆 `agents-ranker`

Finally, the `agents-ranker` tool ranks the agents by simulating an Elo tournament where the output of each game is given by the answers from the `answers-annotator`:

```bash
ragelo agents-ranker answers_eval.jsonl agents_ranking.csv
```
The output of this step is written to the output file `agents_ranking.csv` with columns agent and score: [tests/data/agents_ranking.csv](https://github.com/zetaalphavector/RAGElo/blob/master/tests/data/agents_ranking.csv).


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
- [ ] Add option to few-shot examples
- [ ] Add CI/CD for publishing
- [x] Testing!
- [x] Publish on PyPi
- [x] Add more document evaluators (Microsoft)
- [x] Split Elo evaluator
- [x] Install as standalone CLI

[^1]: The RAGElo logo was created using Dall-E 3 and GPT-4 with the following prompt: "Vector logo design for a toolkit named 'RAGElo'. The logo should have bold, modern typography with emphasis on 'RAG' in a contrasting color. Include a minimalist icon symbolizing retrieval or ranking."
