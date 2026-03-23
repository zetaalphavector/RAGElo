import json

import pytest

from ragelo.types.experiment import Experiment
from ragelo.types.results import RetrievalEvaluationAnswer, RetrievalEvaluatorResult
from ragelo.types.storage import FileStorageBackend, NullStorageBackend, StorageBackend


class TestStorageBackendProtocol:
    def test_file_storage_satisfies_protocol(self, tmp_path):
        backend = FileStorageBackend(tmp_path / "exp.json", tmp_path / "results.jsonl")
        assert isinstance(backend, StorageBackend)

    def test_null_storage_satisfies_protocol(self):
        backend = NullStorageBackend()
        assert isinstance(backend, StorageBackend)


class TestFileStorageBackend:
    @pytest.fixture
    def backend(self, tmp_path):
        return FileStorageBackend(tmp_path / "exp.json", tmp_path / "results.jsonl")

    def test_initialize_creates_files(self, backend):
        backend.initialize()
        assert backend.save_path.exists()
        assert backend.evaluations_cache_path.exists()

    def test_save_and_load_experiment(self, backend):
        backend.initialize()
        data = {"experiment_name": "test", "queries": {"q1": {"query": "hello"}}}
        backend.save_experiment(data)

        loaded = backend.load_experiment()
        assert loaded == data

    def test_load_experiment_returns_none_when_empty(self, backend):
        backend.initialize()
        assert backend.load_experiment() is None

    def test_save_and_load_results(self, backend):
        backend.initialize()
        result = RetrievalEvaluatorResult(
            qid="0",
            did="0",
            evaluator_name="reasoner",
            answer=RetrievalEvaluationAnswer(reasoning="relevant", score=2),
        )
        backend.save_result(result)
        backend.save_result(result)

        lines = list(backend.load_results())
        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["qid"] == "0"
        assert parsed["evaluator_name"] == "reasoner"

    def test_clear_results(self, backend):
        backend.initialize()
        result = RetrievalEvaluatorResult(
            qid="0",
            did="0",
            evaluator_name="reasoner",
            answer=RetrievalEvaluationAnswer(reasoning="relevant", score=2),
        )
        backend.save_result(result)
        backend.clear_results()
        assert list(backend.load_results()) == []

    def test_load_results_no_file(self, tmp_path):
        backend = FileStorageBackend(tmp_path / "exp.json", tmp_path / "no_results.jsonl")
        assert list(backend.load_results()) == []


class TestNullStorageBackend:
    def test_all_ops_are_noop(self):
        backend = NullStorageBackend()
        backend.initialize()
        backend.save_experiment({"foo": "bar"})
        assert backend.load_experiment() is None
        result = RetrievalEvaluatorResult(
            qid="0",
            did="0",
            evaluator_name="reasoner",
            answer=RetrievalEvaluationAnswer(reasoning="relevant", score=2),
        )
        backend.save_result(result)
        assert list(backend.load_results()) == []
        backend.clear_results()


class TestExperimentStorageIntegration:
    def test_experiment_uses_file_storage_by_default(self, tmp_path, base_experiment_config):
        backend = FileStorageBackend(tmp_path / "exp.json", tmp_path / "results.jsonl")
        base_experiment_config["storage_backend"] = backend
        experiment = Experiment(**base_experiment_config)
        assert isinstance(experiment.storage, FileStorageBackend)

    def test_experiment_uses_custom_storage_backend(self, base_experiment_config):
        backend = NullStorageBackend()
        base_experiment_config["storage_backend"] = backend
        experiment = Experiment(**base_experiment_config)
        assert experiment.storage is backend

    def test_save_and_load_with_file_backend(self, tmp_path, base_experiment_config):
        save_path = tmp_path / "exp.json"
        results_path = tmp_path / "exp_results.jsonl"
        base_experiment_config["storage_backend"] = FileStorageBackend(save_path, results_path)
        experiment = Experiment(**base_experiment_config)
        assert len(experiment) == 2

        loaded = Experiment(
            experiment_name="test_experiment",
            storage_backend=FileStorageBackend(save_path, results_path),
        )
        assert len(loaded) == len(experiment)
        for qid in experiment.keys():
            assert loaded[qid].query == experiment[qid].query

    def test_save_result_with_file_backend(
        self,
        tmp_path,
        base_experiment_config,
        retrieval_evaluation,
    ):
        results_path = tmp_path / "results.jsonl"
        base_experiment_config["storage_backend"] = FileStorageBackend(tmp_path / "exp.json", results_path)
        experiment = Experiment(**base_experiment_config)

        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        experiment.add_evaluation((query, doc), retrieval_evaluation, should_save=True)

        with open(results_path) as f:
            lines = f.readlines()
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["evaluator_name"] == "reasoner"

    def test_save_result_with_null_backend(self, base_experiment_config, retrieval_evaluation):
        experiment = Experiment(**base_experiment_config)

        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        # Should not raise — NullStorageBackend is a no-op
        experiment.add_evaluation((query, doc), retrieval_evaluation, should_save=True)
        assert "reasoner" in doc.evaluations
