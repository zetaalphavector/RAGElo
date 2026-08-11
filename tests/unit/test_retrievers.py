import asyncio
import json

import pytest
from pydantic import ValidationError

from ragelo import Document, Experiment, FileRetriever, NamespacedRetriever, RunFile
from ragelo.types.query import Query


def _run_file(path, runs):
    path.write_text(json.dumps({"retriever_name": "baseline", "runs": runs}))
    return path


class TestFileRetriever:
    def test_serves_ranked_documents_and_honours_top_k(self, tmp_path):
        path = _run_file(
            tmp_path / "run.json",
            {
                "q1": [
                    {"did": "d0", "text": "first", "score": 2.0},
                    {"did": "d1", "text": "second", "score": 1.0},
                ]
            },
        )
        retriever = FileRetriever.from_run_file(path)

        top1 = asyncio.run(retriever.retrieve(Query(qid="q1", query="q"), top_k=1))
        assert [document.did for document in top1] == ["d0"]

    def test_a_query_absent_from_the_file_returns_nothing_and_warns(self, tmp_path, caplog):
        path = _run_file(tmp_path / "run.json", {"q1": [{"did": "d0", "text": "t"}]})
        retriever = FileRetriever.from_run_file(path)

        documents = asyncio.run(retriever.retrieve(Query(qid="missing", query="q"), top_k=10))

        assert documents == []
        assert "no entry for query missing" in caplog.text

    def test_drops_textless_entries_and_counts_them(self, tmp_path):
        path = _run_file(tmp_path / "run.json", {"q1": [{"did": "d0", "text": "kept"}, {"did": "d1", "text": ""}]})
        retriever = FileRetriever.from_run_file(path)

        documents = asyncio.run(retriever.retrieve(Query(qid="q1", query="q"), top_k=10))
        assert [document.did for document in documents] == ["d0"]
        assert retriever.skipped_without_text == 1

    def test_a_malformed_run_file_raises_a_validation_error(self, tmp_path):
        path = _run_file(tmp_path / "run.json", {"q1": [{"did": "d0"}]})
        with pytest.raises(ValidationError):
            FileRetriever.from_run_file(path)


class TestNamespacedRetriever:
    def test_prefixes_every_did_and_keeps_the_rest(self, tmp_path):
        path = _run_file(tmp_path / "run.json", {"q1": [{"did": "d0", "text": "t", "score": 5.0}]})
        retriever = NamespacedRetriever(FileRetriever.from_run_file(path), "baseline")

        documents = asyncio.run(retriever.retrieve(Query(qid="q1", query="q"), top_k=10))

        assert [(document.did, document.text, document.score) for document in documents] == [
            ("baseline::d0", "t", 5.0)
        ]


class TestGetRunFiles:
    def test_round_trips_a_pooled_experiment(self, tmp_path):
        experiment = Experiment(experiment_name="t", save_on_disk=False)
        experiment.add_query(Query(qid="q1", query="q"), should_save=False)
        experiment.add_retrieved_doc(Document(qid="q1", did="d0", text="alpha"), score=0.9, agent="sys-a")
        experiment.add_retrieved_doc(Document(qid="q1", did="d1", text="beta"), score=2.5, agent="sys-a")

        run_files = experiment.get_run_files(output_dir=tmp_path)

        dumped = RunFile.model_validate_json((tmp_path / "sys-a.json").read_text())
        assert dumped == run_files["sys-a"]
        assert [document.did for document in dumped.runs["q1"]] == ["d1", "d0"]

        reloaded = FileRetriever.from_run_file(tmp_path / "sys-a.json")
        documents = asyncio.run(reloaded.retrieve(Query(qid="q1", query="q"), top_k=10))
        assert [(document.did, document.text, document.score) for document in documents] == [
            ("d1", "beta", 2.5),
            ("d0", "alpha", 0.9),
        ]

    def test_exports_only_the_requested_agents(self):
        experiment = Experiment(experiment_name="t", save_on_disk=False)
        experiment.add_query(Query(qid="q1", query="q"), should_save=False)
        experiment.add_retrieved_doc(Document(qid="q1", did="d0", text="alpha"), score=1.0, agent="sys-a")
        experiment.add_retrieved_doc(
            Document(qid="q1", did="d0", text="alpha"), score=2.0, agent="sys-b", exist_ok=True
        )

        run_files = experiment.get_run_files(agents=["sys-b"])

        assert list(run_files) == ["sys-b"]
        assert [document.score for document in run_files["sys-b"].runs["q1"]] == [2.0]

    def test_a_reloaded_run_pools_with_the_agent_that_produced_it(self, tmp_path):
        source = Experiment(experiment_name="s", save_on_disk=False)
        source.add_query(Query(qid="q1", query="q"), should_save=False)
        source.add_retrieved_doc(Document(qid="q1", did="d0", text="alpha"), score=1.0, agent="sys-a")
        source.get_run_files(output_dir=tmp_path)

        target = Experiment(experiment_name="t", save_on_disk=False)
        target.add_query(Query(qid="q1", query="q"), should_save=False)
        target.run_retrievers({"sys-a": FileRetriever.from_run_file(tmp_path / "sys-a.json")}, top_k=10)

        assert target["q1"].retrieved_docs["d0"].retrieved_by == {"sys-a": 1.0}
        assert target["q1"].retrieved_docs["d0"].text == "alpha"
