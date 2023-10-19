
<h1 align="center">
<img style="vertical-align:middle" src="docs/images/RAGElo_logo.png" height="200">
</h1>

<p  align="center" >
<i> Elo-based RAG Agent evaluator </i>
</p>



**RAGElo**[^1] is a lightweight yet powerful set of tools for ranking RAG (Retrieval Augmented Generated) LLM agents using the [Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system). RAGElo uses LLMs with battle-proved prompts and methods for calculating a preference ranking between multiple RAG pipelines. It can be used either as a standalone CLI application or integrated into your existing code as a Python library.

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

We need three different files for running the full process, `queries.csv`, `documents.csv` and `answers.csv`. 
There are sample files in the `tests/data` folder. 

Below we describe the format of each file.

`queries.csv`: 
```csv
query_id,query
0, What is the capital of Brazil?
1, What is the capital of France?
```

`documents.csv`:
```csv
query_id,doc_id,document_text
0,0, Brasília is the capital of Brazil.
0,1, Rio de Janeiro used to be the capital of Brazil.
1,2, Paris is the capital of France.
1,3, Lyon is the second largest city in France.
```

`answers.csv`:
```csv
query_id,agent,answer
0, agent1, "Brasília is the capital of Brazil, according to [0]."
0, agent2, "Accodring to [1], Rio de Janeiro used to be the capital of Brazil, until 1960."
1, agent1, "Paris is the capital of France, according to [2]."
1, agent2, "Lyon is the second largest city in France, according to [3]."
```

The OpenAI API key should be set as an environment variable (`OPENAI_API_KEY`). Alternatively, you can set a credentials file and pass it as an option to `ragelo`:

```bash
ragelo --credentials credentials.txt run-all queries.csv documents.csv answers.csv 
```
this file has the following structure:

```
OPENAI_API_KEY=<your_key_here>
```


## 🧩 Components
While **RAGElo** is meant to be used as an end-to-end tool, we can also invoke each of its components individually:

#### 📜 `documents-annotator`
The `documents-annotator` tool annotates a set of documents based on their relevance to the user query. This is done regardless of the answers provided by the Agents. By default, it uses the `reasoner` annotator, that only outputs the reasoning for the relevance judgment:
```bash
ragelo documents-annotator queries.csv documents.csv reasonings.csv reasoner
```
The `reasonings.csv` output file if also a CSV file:
```csv
query_id,did,answer
0,0,"Very relevant: The document directly states that Brasília is the capital of Brazil, answering the user question."
0,1,Very relevant: The document directly answers the user question by stating that Rio de Janeiro used to be the capital of Brazil.
1,2,Very relevant: The document directly answers the user question by stating that Paris is the capital of France.
1,3,Not relevant: The document does not provide information about the capital of France.
```

### 💬 `answers-annotator`

The `answers-annotator` tool annotates the answers generated by the Agents, taking the quality of the documents retrieved by the retrieval pipeline. By default, it uses the `PairwiseWithReasoning` annotator, that generates `k` random pairs of answers for each query and chooses which answer is the best based on the relevance of documents cited in the answer. It relies on the `reasonings.csv` file generated by the `documents-annotator`:

```bash
ragelo answers-annotator queries.csv answers.csv reasonings.csv answers_eval.jsonl
```

The `answers_eval.jsonl` output file is a JSONL file with each line providing the full prompt used for evaluating the pair of answers, the full output of the annotator and the selected answer. An example of the output file is provided at `tests/data/answers_eval.jsonl`.
 
### 🏆 `agents-ranker`

Finally, the `agents-ranker` tool ranks the agents by simulating an Elo tournament where the output of each game is given by the answers from the `answers-annotator`:

```bash
ragelo agents-ranker answers_eval.jsonl agents_ranking.csv
```
The output of this step is written to the output file `agents_ranking.csv` and printed to the console:
```csv
agent,score
agent1,1026.666828786396
agent2,973.3331712136038
```

## 🙋 Contributing

To install the development dependencies, download the repository and run:

```bash
git clone https://github.com/zeta-alpha/ragelo && cd ragelo
pip install -e '.[dev]'
```

This will intstall the requirement dependencies and in an editable mode (i.e., any changes to the code don't need to be re-build.)
For building a new version, use the `build` command:
```bash
python -m build
```

### ✅ TODO
- [ ] Publish on PyPi
- [ ] Add custom types
- [ ] Testing!
- [x] Add more document evaluators (Microsoft)
- [x] Split Elo evaluator
- [x] Install as standalone CLI

[^1]: The RAGElo logo was created using Dall-E 3 and GPT-4 with the following prompt: `Vector logo design for a toolkit named 'RAGElo'. The logo should have bold, modern typography with emphasis on 'RAG' in a contrasting color. Include a minimalist icon symbolizing retrieval or ranking.``
