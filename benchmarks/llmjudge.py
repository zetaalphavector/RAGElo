"""LLMJudge (LLM4Eval, SIGIR 2024): TREC DL 2023 passages with NIST relevance labels on a 0-3 scale."""

from __future__ import annotations

import json
import logging
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ragelo import Experiment

logger = logging.getLogger(__name__)

Split = Literal["dev", "test"]

CHALLENGE_REPO = "https://raw.githubusercontent.com/llm4eval/LLMJudge/81159c68260d383f8da1dad8a14f6d914a9054a9/data"
BENCHMARK_REPO = (
    "https://raw.githubusercontent.com/llm4eval/LLMJudge-benchmark/0e2024814e3192cfc662ecee56dbd16b2536a7de"
    "/llmjudge/qrels"
)
QUERIES_FILE = "llm4eval_query_2024.txt"
PASSAGES_FILE = "llm4eval_document_2024.jsonl"
# The challenge repo ships the test pairs unlabeled. The labeled copy lives in the benchmark repo.
QRELS_FILES: dict[Split, str] = {"dev": "llm4eval_dev_qrel_2024.txt", "test": "llmjudge_test_qrel_idx.txt"}
# The definitions the NIST assessors labeled with, from the LLMJudge challenge README.
GRADES = [
    "Irrelevant: The passage has nothing to do with the query.",
    "Related: The passage seems related to the query but does not answer it.",
    (
        "Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, "
        "or hidden amongst extraneous information."
    ),
    "Perfectly relevant: The passage is dedicated to the query and contains the exact answer.",
]
SOURCES = {
    QUERIES_FILE: CHALLENGE_REPO,
    PASSAGES_FILE: CHALLENGE_REPO,
    QRELS_FILES["dev"]: CHALLENGE_REPO,
    QRELS_FILES["test"]: BENCHMARK_REPO,
}


@dataclass(frozen=True, slots=True)
class LLMJudgeData:
    queries: dict[str, str]
    passages: dict[str, str]
    qrels: dict[str, dict[str, int]]

    @property
    def n_pairs(self) -> int:
        return sum(len(judged) for judged in self.qrels.values())


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for file_name, base_url in SOURCES.items():
        target = data_dir / file_name
        if target.is_file():
            continue
        logger.info(f"Downloading {file_name}")
        urllib.request.urlretrieve(f"{base_url}/{file_name}", target)


def load(data_dir: Path, split: Split) -> LLMJudgeData:
    qrels: dict[str, dict[str, int]] = {}
    with open(data_dir / QRELS_FILES[split], encoding="utf-8") as f:
        for line in f:
            qid, _, did, label = line.split()
            qrels.setdefault(qid, {})[did] = int(label)

    queries = {}
    with open(data_dir / QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t")
            if qid in qrels:
                queries[qid] = text

    judged = {did for labels in qrels.values() for did in labels}
    passages = {}
    with open(data_dir / PASSAGES_FILE, encoding="utf-8") as f:
        for line in f:
            passage = json.loads(line)
            if passage["docid"] in judged:
                passages[passage["docid"]] = passage["doc"]

    return LLMJudgeData(queries=queries, passages=passages, qrels=qrels)


def sample(data: LLMJudgeData, n_pairs: int, seed: int = 42) -> LLMJudgeData:
    """Draws about `n_pairs` pairs, keeping the split's label distribution."""
    pairs_by_label: dict[int, list[tuple[str, str]]] = {}
    for qid, labels in data.qrels.items():
        for did, label in labels.items():
            pairs_by_label.setdefault(label, []).append((qid, did))

    rng = random.Random(seed)
    qrels: dict[str, dict[str, int]] = {}
    for label, pairs in sorted(pairs_by_label.items()):
        for qid, did in rng.sample(pairs, round(n_pairs * len(pairs) / data.n_pairs)):
            qrels.setdefault(qid, {})[did] = label

    judged = {did for labels in qrels.values() for did in labels}
    return LLMJudgeData(
        queries={qid: data.queries[qid] for qid in qrels},
        passages={did: data.passages[did] for did in judged},
        qrels=qrels,
    )


def to_experiment(data: LLMJudgeData, experiment_name: str, save_path: str | None = None) -> Experiment:
    """The human labels stay in `data.qrels`, so nothing an evaluator can render into a prompt carries them."""
    experiment = Experiment(experiment_name, save_path=save_path)
    for qid, text in data.queries.items():
        experiment.add_query(text, query_id=qid, should_save=False)
    for qid, labels in data.qrels.items():
        for did in labels:
            experiment.add_retrieved_doc(
                data.passages[did], query_id=qid, doc_id=did, exist_ok=True, should_save=False
            )
    experiment.save()
    return experiment
