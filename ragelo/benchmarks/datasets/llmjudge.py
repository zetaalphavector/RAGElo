"""LLMJudge (LLM4Eval, SIGIR 2024): TREC DL 2023 passages with NIST relevance labels on a 0-3 scale."""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

from ragelo.benchmarks.datasets.base import DatasetFactory
from ragelo.benchmarks.datasets.retrieval import RetrievalData, RetrievalDataset
from ragelo.types.types import BenchmarkDatasetTypes

logger = logging.getLogger(__name__)

CHALLENGE_REPO = "https://raw.githubusercontent.com/llm4eval/LLMJudge/81159c68260d383f8da1dad8a14f6d914a9054a9/data"
BENCHMARK_REPO = (
    "https://raw.githubusercontent.com/llm4eval/LLMJudge-benchmark/0e2024814e3192cfc662ecee56dbd16b2536a7de"
    "/llmjudge/qrels"
)
QUERIES_FILE = "llm4eval_query_2024.txt"
PASSAGES_FILE = "llm4eval_document_2024.jsonl"
# The challenge repo ships the test pairs unlabeled. The labeled copy lives in the benchmark repo.
QRELS_FILES = {"dev": "llm4eval_dev_qrel_2024.txt", "test": "llmjudge_test_qrel_idx.txt"}
SOURCES = {
    QUERIES_FILE: CHALLENGE_REPO,
    PASSAGES_FILE: CHALLENGE_REPO,
    QRELS_FILES["dev"]: CHALLENGE_REPO,
    QRELS_FILES["test"]: BENCHMARK_REPO,
}


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for file_name, base_url in SOURCES.items():
        target = data_dir / file_name
        if target.is_file():
            continue
        logger.info(f"Downloading {file_name}")
        urllib.request.urlretrieve(f"{base_url}/{file_name}", target)


def load(data_dir: Path, split: str) -> RetrievalData:
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

    return RetrievalData(queries=queries, passages=passages, qrels=qrels)


@DatasetFactory.register(BenchmarkDatasetTypes.LLMJUDGE)
class LLMJudgeDataset(RetrievalDataset):
    data_name = "llmjudge"
    splits = ("dev", "test")

    def download(self) -> None:
        download(self.data_dir)

    def load(self) -> RetrievalData:
        return load(self.data_dir, self.split)
