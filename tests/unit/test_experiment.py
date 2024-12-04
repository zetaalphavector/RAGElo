from __future__ import annotations

import json
import os

import pytest

from ragelo.types.evaluables import AgentAnswer, Document
from ragelo.types.experiment import Experiment
from ragelo.types.query import Query
from ragelo.types.results import AnswerEvaluatorResult, RetrievalEvaluatorResult


class TestExperiment:
    def test_experiment_initialization(self, experiment):
        assert len(experiment) == 2
        assert "0" in experiment.keys()
        assert "1" in experiment.keys()

        # Check queries were loaded correctly
        assert experiment["0"].query == "What is the capital of Brazil?"
        assert experiment["1"].query == "What is the capital of France?"

        # Check documents were loaded correctly
        assert len(experiment["0"].retrieved_docs) == 2
        assert experiment["0"].retrieved_docs["0"].text == "Brasília is the capital of Brazil."

        # Check answers were loaded correctly
        assert len(experiment["0"].answers) == 2
        assert "agent1" in experiment["0"].answers
        assert "agent2" in experiment["0"].answers

    def test_add_query(self, empty_experiment):
        """Test adding queries manually"""
        # Add query as string
        qid = empty_experiment.add_query("Test query 1", query_id="test1")
        assert qid in empty_experiment.queries
        assert empty_experiment[qid].query == "Test query 1"

        # Add query as Query object
        query = Query(qid="test2", query="Test query 2")
        qid2 = empty_experiment.add_query(query)
        assert qid2 == "test2"
        assert empty_experiment[qid2].query == "Test query 2"

        # Test force parameter
        with pytest.warns(UserWarning):
            empty_experiment.add_query("New query", qid, force=False)
        empty_experiment.add_query("Forced query", qid, force=True)
        assert empty_experiment[qid].query == "Forced query"

    def test_add_retrieved_doc(self, empty_experiment):
        """Test adding retrieved documents manually"""
        qid = empty_experiment.add_query("Test query", query_id="test1")

        # Add document as string
        empty_experiment.add_retrieved_doc("Test document", query_id=qid, doc_id="doc1", score=0.5, agent="agent1")
        assert "doc1" in empty_experiment[qid].retrieved_docs
        assert empty_experiment[qid].retrieved_docs["doc1"].text == "Test document"

        # Add document as Document object
        doc = Document(qid=qid, did="doc2", text="Test document 2")
        empty_experiment.add_retrieved_doc(doc, agent="agent2")
        assert "doc2" in empty_experiment[qid].retrieved_docs

        # Test adding to non-existent query
        with pytest.raises(ValueError):
            empty_experiment.add_retrieved_doc("Test", "invalid_qid", "doc3")

    def test_add_agent_answer(self, empty_experiment):
        """Test adding agent answers manually"""
        qid = empty_experiment.add_query("Test query", query_id="test1")

        # Add answer as string
        empty_experiment.add_agent_answer("Test answer", agent="agent1", query_id=qid)
        assert "agent1" in empty_experiment[qid].answers
        assert empty_experiment[qid].answers["agent1"].text == "Test answer"

        # Add answer as AgentAnswer object
        answer = AgentAnswer(qid=qid, agent="agent2", text="Test answer 2")
        empty_experiment.add_agent_answer(answer)
        assert "agent2" in empty_experiment[qid].answers

        # Test adding to non-existent query
        with pytest.raises(ValueError):
            empty_experiment.add_agent_answer("Test", "agent3", "invalid_qid")

    def test_save_and_load(self, tmp_path, base_experiment_config):
        """Test saving and loading experiment state"""
        # Save experiment
        save_path = tmp_path / "test_experiment.json"
        base_experiment_config["save_on_disk"] = True
        base_experiment_config["save_path"] = str(save_path)
        experiment = Experiment(**base_experiment_config)
        experiment.save()

        # Load experiment
        loaded_experiment = Experiment(
            experiment_name="test_experiment",
            save_path=str(save_path),
            cache_evaluations=False,
            save_on_disk=True,
        )

        # Verify contents
        assert len(loaded_experiment) == len(experiment)
        assert list(loaded_experiment.keys()) == list(experiment.keys())
        for qid in experiment.keys():
            assert loaded_experiment[qid].query == experiment[qid].query
            assert len(loaded_experiment[qid].retrieved_docs) == len(experiment[qid].retrieved_docs)
            assert len(loaded_experiment[qid].answers) == len(experiment[qid].answers)

    def test_get_qrels(self, tmp_path, experiment):
        """Test getting relevance judgments"""
        qrels = experiment.get_qrels()

        # Basic structure check
        assert len(qrels) == 2  # Two queries
        assert "0" in qrels
        assert "1" in qrels

        # Test saving qrels
        output_path = tmp_path / "test_qrels.txt"
        experiment.get_qrels(output_path=str(output_path), output_format="trec")
        assert os.path.exists(output_path)

    def test_add_retrieval_evaluation(self, experiment, retrieval_evaluation, caplog):
        """Test adding retrieval evaluation"""
        # Add evaluation
        experiment.add_evaluation(retrieval_evaluation, should_save=False)

        # Verify evaluation was added
        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        assert doc.evaluation is not None
        assert doc.evaluation.answer == 1
        assert doc.evaluation.raw_answer == "Document is relevant. Score: 1.0"

        # Test adding duplicate evaluation without force
        added = experiment.add_evaluation(retrieval_evaluation, should_save=False)
        assert "Document 0 in query 0 already has an evaluation." in caplog.text
        assert not added

        # Test adding duplicate evaluation with force
        modified_eval = RetrievalEvaluatorResult(qid="0", did="0", raw_answer="Modified evaluation", answer=2)
        experiment.add_evaluation(modified_eval, should_save=False, force=True)
        assert doc.evaluation.answer == 2

        # Test adding evaluation for non-existent query
        invalid_eval = RetrievalEvaluatorResult(qid="999", did="0", raw_answer="Invalid", answer=1)
        with pytest.raises(ValueError):
            experiment.add_evaluation(invalid_eval)

    def test_add_answer_evaluation(self, experiment, answer_evaluation, caplog):
        """Test adding answer evaluation"""
        # Add evaluation
        experiment.add_evaluation(answer_evaluation, should_save=False)

        # Verify evaluation was added
        query = experiment["0"]
        answer = query.answers["agent1"]
        assert answer.evaluation is not None
        assert answer.evaluation.answer == {"quality": 1.0, "relevance": 0.8}
        assert answer.evaluation.raw_answer == "Answer is good. Scores: {'quality': 1.0, 'relevance': 0.8}"
        assert not answer.evaluation.pairwise

        # Test adding duplicate evaluation without force
        added = experiment.add_evaluation(answer_evaluation, should_save=False)
        assert not added
        assert "Agent agent1 in query 0 already has an evaluation." in caplog.text

        # Test adding duplicate evaluation with force
        modified_eval = AnswerEvaluatorResult(
            qid="0",
            agent="agent1",
            pairwise=False,
            raw_answer="Modified evaluation",
            answer={"quality": 2.0, "relevance": 1.0},
        )
        experiment.add_evaluation(modified_eval, should_save=False, force=True)
        assert answer.evaluation.answer == {"quality": 2.0, "relevance": 1.0}

    def test_add_pairwise_evaluation(self, experiment, pairwise_answer_evaluation):
        """Test adding pairwise answer evaluation"""
        # Add evaluation
        experiment.add_evaluation(pairwise_answer_evaluation, should_save=False)

        # Verify evaluation was added
        query = experiment["0"]
        assert len(query.pairwise_games) == 1
        game = query.pairwise_games[0]
        assert game.evaluation is not None
        assert game.evaluation.answer == "A"
        assert game.evaluation.raw_answer == "Answer [[A]] is better than [[B]]"
        assert game.evaluation.pairwise
        assert game.evaluation.agent_a == "agent1"
        assert game.evaluation.agent_b == "agent2"

    def test_add_elo_tournament(self, experiment, elo_tournament_result):
        """Test adding Elo tournament results"""
        # Add evaluation
        experiment.add_evaluation(elo_tournament_result, should_save=False)

        # Verify tournament was added
        assert len(experiment.elo_tournaments) == 1
        tournament = experiment.elo_tournaments[0]

        # Check tournament results
        assert tournament.agents == ["agent1", "agent2"]
        assert tournament.scores == {"agent1": 1200, "agent2": 1000}
        assert tournament.games_played == {"agent1": 1, "agent2": 1}
        assert tournament.wins == {"agent1": 1, "agent2": 0}
        assert tournament.loses == {"agent1": 0, "agent2": 1}
        assert tournament.ties == {"agent1": 0, "agent2": 0}
        assert tournament.total_games == 2
        assert tournament.total_tournaments == 1

    def test_save_evaluations(
        self,
        tmp_path,
        base_experiment_config,
        retrieval_evaluation,
        answer_evaluation,
        pairwise_answer_evaluation,
        elo_tournament_result,
    ):
        """Test saving evaluations to disk"""
        # Set up save paths
        results_path = tmp_path / "test_results.jsonl"
        base_experiment_config["evaluations_cache_path"] = str(results_path)
        base_experiment_config["cache_evaluations"] = True
        experiment = Experiment(**base_experiment_config)

        # Add and save evaluations
        experiment.add_evaluation(retrieval_evaluation, should_save=True)
        experiment.add_evaluation(answer_evaluation, should_save=True)
        experiment.add_evaluation(pairwise_answer_evaluation, should_save=True)
        experiment.add_evaluation(elo_tournament_result, should_save=True)

        # Verify results were saved
        assert results_path.exists()

        # Load and verify saved results
        with open(results_path) as f:
            lines = f.readlines()
            assert len(lines) == 4

            # Parse each line and verify content types
            results = [json.loads(line) for line in lines]
            result_types = [list(r.keys())[0] for r in results]
            assert "retrieval" in result_types
            assert "answer" in result_types
            assert "elo_tournament" in result_types

    def test_clear_evaluations(
        self,
        experiment,
        retrieval_evaluation,
        answer_evaluation,
        pairwise_answer_evaluation,
        elo_tournament_result,
    ):
        """Test clearing all evaluations"""
        # Add evaluations
        experiment.add_evaluation(retrieval_evaluation, should_save=False)
        experiment.add_evaluation(answer_evaluation, should_save=False)
        experiment.add_evaluation(pairwise_answer_evaluation, should_save=False)
        experiment.add_evaluation(elo_tournament_result, should_save=False)

        # Clear evaluations
        experiment._Experiment__clear_all_evaluations()

        # Verify evaluations were cleared
        query = experiment["0"]
        assert query.retrieved_docs["0"].evaluation is None
        assert query.answers["agent1"].evaluation is None
        assert len(experiment.elo_tournaments) == 0

    def test_get_runs(self, tmp_path, experiment_with_retrieval_scores):
        """Test getting retrieval runs"""
        runs = experiment_with_retrieval_scores.get_runs()

        # Basic structure check
        assert len(runs) > 0  # At least one agent

        # Test saving runs
        output_dir = tmp_path / "test_runs"
        experiment_with_retrieval_scores.get_runs(output_path=str(output_dir), output_format="trec")
        assert os.path.exists(output_dir)
        import shutil

        shutil.rmtree(output_dir)  # Cleanup

    def test_iteration(self, experiment):
        """Test iteration over queries"""
        queries = list(experiment)
        assert len(queries) == 2
        assert all(isinstance(q, Query) for q in queries)

    def test_add_queries_from_csv(self, empty_experiment):
        """Test adding queries from CSV"""
        empty_experiment.add_queries_from_csv("tests/data/queries.csv")
        assert len(empty_experiment) == 2
        assert "0" in empty_experiment.keys()
        assert "1" in empty_experiment.keys()

    def test_add_documents_from_csv(self, empty_experiment):
        """Test adding documents from CSV"""
        empty_experiment.add_queries_from_csv("tests/data/queries.csv")
        empty_experiment.add_documents_from_csv("tests/data/documents.csv")

        assert len(empty_experiment["0"].retrieved_docs) == 2
        assert len(empty_experiment["1"].retrieved_docs) == 2

    def test_add_agent_answers_from_csv(self, empty_experiment):
        """Test adding answers from CSV"""
        empty_experiment.add_queries_from_csv("tests/data/queries.csv")
        empty_experiment.add_agent_answers_from_csv("tests/data/answers.csv")

        assert len(empty_experiment["0"].answers) == 2
        assert len(empty_experiment["1"].answers) == 2
