import json
import os
import shutil

import pytest

from ragelo import Experiment, get_agent_ranker, get_answer_evaluator, get_llm_provider, get_retrieval_evaluator
from ragelo.types.evaluables import AgentAnswer, Document
from ragelo.types.query import Query
from ragelo.types.results import (
    AnswerEvaluationAnswer,
    AnswerEvaluatorResult,
    PairwiseEvaluationAnswer,
    PairwiseGameEvaluatorResult,
    RetrievalEvaluationAnswer,
    RetrievalEvaluatorResult,
)


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

    def test_add_retrieved_doc_preserves_retrieved_by(self, empty_experiment):
        """Test that re-adding a document merges retrieved_by info."""
        qid = empty_experiment.add_query("Test query", query_id="test1")
        empty_experiment.add_retrieved_doc("Test doc", query_id=qid, doc_id="doc1", score=1.0, agent="agent1")
        assert empty_experiment[qid].retrieved_docs["doc1"].retrieved_by == {"agent1": 1.0}

        # Re-add same doc with a different agent — retrieved_by should merge
        doc = Document(qid=qid, did="doc1", text="Test doc")
        doc.retrieved_by = {"agent2": 0.5}
        empty_experiment[qid].add_retrieved_doc(doc, agent="agent2")
        assert "agent1" in empty_experiment[qid].retrieved_docs["doc1"].retrieved_by
        assert "agent2" in empty_experiment[qid].retrieved_docs["doc1"].retrieved_by
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
            save_on_disk=True,
        )

        # Verify contents
        assert len(loaded_experiment) == len(experiment)
        assert list(loaded_experiment.keys()) == list(experiment.keys())
        for qid in experiment.keys():
            assert loaded_experiment[qid].query == experiment[qid].query
            assert len(loaded_experiment[qid].retrieved_docs) == len(experiment[qid].retrieved_docs)
            assert len(loaded_experiment[qid].answers) == len(experiment[qid].answers)
            for did, doc in experiment[qid].retrieved_docs.items():
                assert loaded_experiment[qid].retrieved_docs[did].retrieved_by == doc.retrieved_by

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
        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        experiment.add_evaluation((query, doc), retrieval_evaluation, should_save=False)

        # Verify evaluation was added
        assert len(doc.evaluations) > 0
        assert "reasoner" in doc.evaluations
        evaluation = doc.evaluations["reasoner"]
        assert evaluation.answer.score == 1.0  # Access nested answer
        assert evaluation.answer.reasoning == "The document is relevant"

        # Test adding duplicate evaluation without force
        added = experiment.add_evaluation((query, doc), retrieval_evaluation, should_save=False)
        assert "Evaluable 0 in query 0 already has an evaluation" in caplog.text
        assert not added

        # Test adding duplicate evaluation with force
        modified_eval = RetrievalEvaluatorResult(
            qid="0",
            did="0",
            evaluator_name="reasoner",
            answer=RetrievalEvaluationAnswer(reasoning="Modified", score=2.0),
        )
        experiment.add_evaluation((query, doc), modified_eval, should_save=False, force=True)
        assert doc.evaluations["reasoner"].answer.score == 2.0

        # Test adding evaluation for non-existent query
        invalid_eval = RetrievalEvaluatorResult(
            qid="999",
            did="0",
            evaluator_name="reasoner",
            answer=RetrievalEvaluationAnswer(reasoning="Invalid", score=1.0),
        )
        with pytest.raises(ValueError):
            experiment.add_evaluation(None, invalid_eval)

    def test_add_answer_evaluation(self, experiment, answer_evaluation, caplog):
        """Test adding answer evaluation"""

        # Add evaluation
        query = experiment["0"]
        answer = query.answers["agent1"]
        experiment.add_evaluation((query, answer), answer_evaluation, should_save=False)

        # Verify evaluation was added
        assert len(answer.evaluations) > 0
        assert "custom_prompt" in answer.evaluations
        evaluation = answer.evaluations["custom_prompt"]
        assert evaluation.answer.score == 1  # Access nested answer
        assert evaluation.answer.reasoning == "Good quality answer"

        # Test adding duplicate evaluation without force
        added = experiment.add_evaluation((query, answer), answer_evaluation, should_save=False)
        assert not added
        assert "Evaluable agent1 in query 0 already has an evaluation" in caplog.text

        # Test adding duplicate evaluation with force
        modified_eval = AnswerEvaluatorResult(
            qid="0",
            agent="agent1",
            evaluator_name="custom_prompt",
            answer=AnswerEvaluationAnswer(reasoning="Modified quality", score=2),
        )
        experiment.add_evaluation((query, answer), modified_eval, should_save=False, force=True)
        assert answer.evaluations["custom_prompt"].answer.score == 2

    def test_add_pairwise_evaluation(self, experiment, pairwise_answer_evaluation):
        """Test adding pairwise answer evaluation"""
        # Add evaluation
        query = experiment["0"]
        # First, we need to create the pairwise game
        game = query.add_pairwise_game("agent1", "agent2")
        experiment.add_evaluation((query, game), pairwise_answer_evaluation, should_save=False)

        # Verify evaluation was added
        assert len(query.pairwise_games) >= 1
        assert len(game.evaluations) > 0
        assert "pairwise" in game.evaluations
        evaluation = game.evaluations["pairwise"]
        assert evaluation.answer.winner == "A"  # Access nested answer
        assert evaluation.answer.answer_a_analysis == "Answer A is good"
        assert evaluation.answer.answer_b_analysis == "Answer B is less good"
        assert evaluation.answer.comparison_reasoning == "A is better"
        assert evaluation.agent_a == "agent1"
        assert evaluation.agent_b == "agent2"

    def test_add_elo_tournament(self, experiment, elo_tournament_result):
        """Test adding Elo tournament results"""
        # Add evaluation (EloTournamentResult doesn't need eval_tuple)
        experiment.add_evaluation(None, elo_tournament_result, should_save=False)

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
        base_experiment_config["save_on_disk"] = True
        experiment = Experiment(**base_experiment_config)

        # Add and save evaluations
        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        answer = query.answers["agent1"]
        game = query.add_pairwise_game("agent1", "agent2")

        experiment.add_evaluation((query, doc), retrieval_evaluation, should_save=True)
        experiment.add_evaluation((query, answer), answer_evaluation, should_save=True)
        experiment.add_evaluation((query, game), pairwise_answer_evaluation, should_save=True)
        experiment.add_evaluation(None, elo_tournament_result, should_save=True)

        # Verify results were saved
        assert results_path.exists()

        # Load and verify saved results
        with open(results_path) as f:
            lines = f.readlines()
            assert len(lines) == 4

            # Parse each line and verify they have proper structure
            results = [json.loads(line) for line in lines]
            # Check that results have evaluator_name and answer fields
            for result in results:
                # EloTournament doesn't have evaluator_name
                assert "evaluator_name" in result or "total_games" in result
                # Verify answer is an object (nested) for non-Elo results
                if "answer" in result:
                    assert isinstance(result["answer"], dict), "Answer should be serialized as dict"

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
        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        answer = query.answers["agent1"]
        game = query.add_pairwise_game("agent1", "agent2")

        experiment.add_evaluation((query, doc), retrieval_evaluation, should_save=False)
        experiment.add_evaluation((query, answer), answer_evaluation, should_save=False)
        experiment.add_evaluation((query, game), pairwise_answer_evaluation, should_save=False)
        experiment.add_evaluation(None, elo_tournament_result, should_save=False)

        # Clear evaluations
        experiment._Experiment__clear_all_evaluations()

        # Verify evaluations were cleared
        assert len(query.retrieved_docs["0"].evaluations) == 0
        assert len(query.answers["agent1"].evaluations) == 0
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

    @pytest.mark.requires_openai
    def test_readme_example(self):
        """Test the README example"""
        if os.path.exists("ragelo_cache/A_really_cool_RAGElo_experiment.json"):
            os.remove("ragelo_cache/A_really_cool_RAGElo_experiment.json")
        if os.path.exists("ragelo_cache/A_really_cool_RAGElo_experiment_results.jsonl"):
            os.remove("ragelo_cache/A_really_cool_RAGElo_experiment_results.jsonl")

        experiment = Experiment(experiment_name="A_really_cool_RAGElo_experiment")
        # Add two user queries. Alternatively, we can load them from a csv file with .add_queries_from_csv()
        experiment.add_query("What is the capital of Brazil?", query_id="q0")
        experiment.add_query("What is the capital of France?", query_id="q1")

        # Add four documents retrieved for these queries.
        # Alternatively, we can load them from a csv file with .add_documents_from_csv()
        experiment.add_retrieved_doc("Brasília is the capital of Brazil", query_id="q0", doc_id="d0")
        experiment.add_retrieved_doc("Rio de Janeiro used to be the capital of Brazil.", query_id="q0", doc_id="d1")
        experiment.add_retrieved_doc("Paris is the capital of France.", query_id="q1", doc_id="d2")
        experiment.add_retrieved_doc("Lyon is the second largest city in France.", query_id="q1", doc_id="d3")

        # Add the answers generated by agents
        experiment.add_agent_answer(
            "Brasília is the capital of Brazil, according to [0].", agent="agent1", query_id="q0"
        )
        experiment.add_agent_answer(
            "According to [1], Rio de Janeiro used to be the capital of Brazil, until the 60s.",
            agent="agent2",
            query_id="q0",
        )
        experiment.add_agent_answer("Paris is the capital of France, according to [2].", agent="agent1", query_id="q1")
        experiment.add_agent_answer(
            "According to [3], Lyon is the second largest city in France. Meanwhile, Paris is its capital [2].",
            agent="agent2",
            query_id="q1",
        )

        llm_provider = get_llm_provider("openai", model="gpt-4.1-nano")

        retrieval_evaluator = get_retrieval_evaluator("reasoner", llm_provider=llm_provider, rich_print=True)
        answer_evaluator = get_answer_evaluator("pairwise", llm_provider=llm_provider, rich_print=True)

        elo_ranker = get_agent_ranker("elo", show_results=True)

        # Evaluate the retrieval results.
        retrieval_evaluator.evaluate_experiment(experiment)

        # With the retrieved documents evaluated, evaluate the quality of the answers. using the pairwise evaluator
        answer_evaluator.evaluate_experiment(experiment)
        result = elo_ranker.run(experiment)

        # Verify the Elo tournament produced valid results for both agents
        assert "agent1" in result.scores
        assert "agent2" in result.scores
        assert result.total_games > 0
        assert result.total_tournaments > 0

        assert os.path.exists("ragelo_cache/A_really_cool_RAGElo_experiment.json")
        assert os.path.exists("ragelo_cache/A_really_cool_RAGElo_experiment_results.jsonl")
        os.remove("ragelo_cache/A_really_cool_RAGElo_experiment.json")
        os.remove("ragelo_cache/A_really_cool_RAGElo_experiment_results.jsonl")


class TestExperimentSerialization:
    """Tests for experiment serialization and deserialization with nested answer schemas."""

    def test_round_trip_retrieval_evaluation(self, tmp_path, retrieval_evaluation):
        """Test round-trip serialization for retrieval evaluation."""
        # Save result to JSONL
        results_path = tmp_path / "test_results.jsonl"
        results_path.touch()

        # Serialize
        with open(results_path, "w") as f:
            f.write(retrieval_evaluation.model_dump_json() + "\n")

        # Deserialize
        with open(results_path) as f:
            loaded_data = json.loads(f.readline())
            loaded_result = RetrievalEvaluatorResult.model_validate(loaded_data)

        # Verify nested answer is properly deserialized
        assert isinstance(loaded_result.answer, RetrievalEvaluationAnswer)
        assert loaded_result.answer.score == retrieval_evaluation.answer.score
        assert loaded_result.answer.reasoning == retrieval_evaluation.answer.reasoning
        assert loaded_result.qid == retrieval_evaluation.qid
        assert loaded_result.did == retrieval_evaluation.did
        assert loaded_result.evaluator_name == retrieval_evaluation.evaluator_name

    def test_round_trip_answer_evaluation(self, tmp_path, answer_evaluation):
        """Test round-trip serialization for answer evaluation."""
        # Save result to JSONL
        results_path = tmp_path / "test_results.jsonl"
        results_path.touch()

        # Serialize
        with open(results_path, "w") as f:
            f.write(answer_evaluation.model_dump_json() + "\n")

        # Deserialize
        with open(results_path) as f:
            loaded_data = json.loads(f.readline())
            loaded_result = AnswerEvaluatorResult.model_validate(loaded_data)

        # Verify nested answer is properly deserialized
        assert isinstance(loaded_result.answer, AnswerEvaluationAnswer)
        assert loaded_result.answer.score == answer_evaluation.answer.score
        assert loaded_result.answer.reasoning == answer_evaluation.answer.reasoning
        assert loaded_result.qid == answer_evaluation.qid
        assert loaded_result.agent == answer_evaluation.agent

    def test_round_trip_pairwise_evaluation(self, tmp_path, pairwise_answer_evaluation):
        """Test round-trip serialization for pairwise evaluation."""
        # Save result to JSONL
        results_path = tmp_path / "test_results.jsonl"
        results_path.touch()

        # Serialize
        with open(results_path, "w") as f:
            f.write(pairwise_answer_evaluation.model_dump_json() + "\n")

        # Deserialize
        with open(results_path) as f:
            loaded_data = json.loads(f.readline())
            loaded_result = PairwiseGameEvaluatorResult.model_validate(loaded_data)

        # Verify nested answer is properly deserialized
        assert isinstance(loaded_result.answer, PairwiseEvaluationAnswer)
        assert loaded_result.answer.winner == pairwise_answer_evaluation.answer.winner
        assert loaded_result.answer.answer_a_analysis == pairwise_answer_evaluation.answer.answer_a_analysis
        assert loaded_result.answer.answer_b_analysis == pairwise_answer_evaluation.answer.answer_b_analysis
        assert loaded_result.answer.comparison_reasoning == pairwise_answer_evaluation.answer.comparison_reasoning

    def test_experiment_with_evaluations_round_trip(
        self, tmp_path, retrieval_evaluation, answer_evaluation, pairwise_answer_evaluation
    ):
        """Test saving and loading experiment with evaluations."""
        # Create experiment with save enabled using unique temp paths
        save_path = tmp_path / "unique_experiment_roundtrip.json"
        results_path = tmp_path / "unique_results_roundtrip.jsonl"

        experiment = Experiment(
            experiment_name="unique_test_round_trip",
            save_on_disk=True,
            save_path=str(save_path),
            evaluations_cache_path=str(results_path),
            queries_csv_path="tests/data/queries.csv",
            documents_csv_path="tests/data/documents.csv",
            answers_csv_path="tests/data/answers.csv",
        )

        # Add evaluations
        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        answer = query.answers["agent1"]
        game = query.add_pairwise_game("agent1", "agent2")

        experiment.add_evaluation((query, doc), retrieval_evaluation, should_save=True)
        experiment.add_evaluation((query, answer), answer_evaluation, should_save=True)
        experiment.add_evaluation((query, game), pairwise_answer_evaluation, should_save=True)

        # Save experiment
        experiment.save()

        # Load experiment in a new instance
        loaded_experiment = Experiment(
            experiment_name="unique_test_round_trip",
            save_on_disk=True,
            save_path=str(save_path),
            evaluations_cache_path=str(results_path),
        )

        # Verify evaluations were loaded
        loaded_query = loaded_experiment["0"]
        loaded_doc = loaded_query.retrieved_docs["0"]
        loaded_answer = loaded_query.answers["agent1"]

        # Check retrieval evaluation
        assert len(loaded_doc.evaluations) > 0
        assert "reasoner" in loaded_doc.evaluations
        assert isinstance(loaded_doc.evaluations["reasoner"], RetrievalEvaluatorResult)
        assert isinstance(loaded_doc.evaluations["reasoner"].answer, RetrievalEvaluationAnswer)
        assert loaded_doc.evaluations["reasoner"].answer.score == 1.0
        assert loaded_doc.evaluations["reasoner"].answer.reasoning == "The document is relevant"

        # Check answer evaluation
        assert len(loaded_answer.evaluations) > 0
        assert "custom_prompt" in loaded_answer.evaluations
        assert isinstance(loaded_answer.evaluations["custom_prompt"], AnswerEvaluatorResult)
        assert isinstance(loaded_answer.evaluations["custom_prompt"].answer, AnswerEvaluationAnswer)
        assert loaded_answer.evaluations["custom_prompt"].answer.score == 1
        assert loaded_answer.evaluations["custom_prompt"].answer.reasoning == "Good quality answer"

        # Check pairwise evaluation
        loaded_game = loaded_query.pairwise_games.get("agent1-agent2")
        assert loaded_game is not None
        assert len(loaded_game.evaluations) > 0
        assert "pairwise" in loaded_game.evaluations
        assert isinstance(loaded_game.evaluations["pairwise"], PairwiseGameEvaluatorResult)
        assert isinstance(loaded_game.evaluations["pairwise"].answer, PairwiseEvaluationAnswer)
        assert loaded_game.evaluations["pairwise"].answer.winner == "A"

    def test_convenience_properties_after_deserialization(self, tmp_path, retrieval_evaluation):
        """Test that convenience properties work after deserialization."""
        # Serialize and deserialize
        results_path = tmp_path / "test_results.jsonl"
        with open(results_path, "w") as f:
            f.write(retrieval_evaluation.model_dump_json() + "\n")

        with open(results_path) as f:
            loaded_data = json.loads(f.readline())
            loaded_result = RetrievalEvaluatorResult.model_validate(loaded_data)

        # Verify convenience properties work (backward compatibility)
        assert loaded_result.answer is not None
        assert hasattr(loaded_result.answer, "score")
        assert hasattr(loaded_result.answer, "reasoning")
        assert loaded_result.score == loaded_result.answer.score
        assert loaded_result.reasoning == loaded_result.answer.reasoning
        assert loaded_result.score == 1.0
        assert loaded_result.reasoning == "The document is relevant"

    def test_serialized_answer_is_nested_dict(
        self, retrieval_evaluation, answer_evaluation, pairwise_answer_evaluation
    ):
        """Test that serialized answer field is a nested dictionary."""
        # Test retrieval evaluation
        retrieval_dict = json.loads(retrieval_evaluation.model_dump_json())
        assert "answer" in retrieval_dict
        assert isinstance(retrieval_dict["answer"], dict)
        assert "score" in retrieval_dict["answer"]
        assert "reasoning" in retrieval_dict["answer"]

        # Test answer evaluation
        answer_dict = json.loads(answer_evaluation.model_dump_json())
        assert "answer" in answer_dict
        assert isinstance(answer_dict["answer"], dict)
        assert "score" in answer_dict["answer"]
        assert "reasoning" in answer_dict["answer"]

        # Test pairwise evaluation
        pairwise_dict = json.loads(pairwise_answer_evaluation.model_dump_json())
        assert "answer" in pairwise_dict
        assert isinstance(pairwise_dict["answer"], dict)
        assert "winner" in pairwise_dict["answer"]
        assert "answer_a_analysis" in pairwise_dict["answer"]
        assert "answer_b_analysis" in pairwise_dict["answer"]

    def test_load_corrupted_jsonl_gracefully(self, tmp_path, base_experiment_config):
        """Test that experiment handles corrupted JSONL gracefully."""
        results_path = tmp_path / "corrupted.jsonl"
        base_experiment_config["save_on_disk"] = True
        base_experiment_config["evaluations_cache_path"] = str(results_path)

        # Create corrupted JSONL file with various issues
        with open(results_path, "w") as f:
            # Invalid JSON
            f.write("This is not valid JSON\n")
            # Missing evaluator_name
            f.write('{"qid": "0", "did": "0", "answer": {"score": 1}}\n')
            # Valid entry
            valid_result = RetrievalEvaluatorResult(
                qid="0",
                did="0",
                evaluator_name="reasoner",
                answer=RetrievalEvaluationAnswer(reasoning="Good", score=2.0),
            )
            f.write(valid_result.model_dump_json() + "\n")

        # Should not crash when loading
        experiment = Experiment(**base_experiment_config)

        # Should have loaded the valid result and skipped the corrupted ones
        if len(experiment.queries) > 0 and "0" in experiment.queries:
            query = experiment["0"]
            if "0" in query.retrieved_docs:
                # The valid result should have been loaded
                doc = query.retrieved_docs["0"]
                if len(doc.evaluations) > 0:
                    assert "reasoner" in doc.evaluations

    def test_missing_evaluable_in_loaded_results(self, tmp_path, base_experiment_config):
        """Test that experiment handles results for missing documents/answers gracefully."""
        results_path = tmp_path / "orphaned_results.jsonl"
        base_experiment_config["save_on_disk"] = True
        base_experiment_config["evaluations_cache_path"] = str(results_path)

        # Create result for non-existent document
        with open(results_path, "w") as f:
            orphaned_result = RetrievalEvaluatorResult(
                qid="0",
                did="999",  # This document doesn't exist
                evaluator_name="reasoner",
                answer=RetrievalEvaluationAnswer(reasoning="Orphaned", score=1.0),
            )
            f.write(orphaned_result.model_dump_json() + "\n")

        # Should not crash when loading, should just skip the orphaned result
        experiment = Experiment(**base_experiment_config)

        # Verify no crash occurred and queries were loaded
        assert len(experiment) == 2
