from __future__ import annotations

from typing import Dict

import rich

from ragelo.types.results import (
    AnswerEvaluatorResult,
    EloTournamentResult,
    PairwiseGameEvaluatorResult,
    RetrievalEvaluatorResult,
)


def _render_elo_tournament(evaluation: EloTournamentResult, rich_print: bool = True):
    scores = sorted(evaluation.scores.items(), key=lambda x: x[1], reverse=True)
    if rich_print:
        rich.print("-------[bold white] Agents Elo Ratings [/bold white]-------")
    else:
        print("------- Agents Elo Ratings -------")
    for _agent, rating in scores:
        std_dev = evaluation.std_dev.get(_agent, 0)
        if rich_print:
            rich.print(f"[bold white]{_agent:<15}[/bold white]: {rating:.1f}(±{std_dev:.1f})")
        else:
            print(f"{_agent:<15}: {rating:.1f} (±{std_dev:.1f})")


def _render_answer_evaluation(evaluation: AnswerEvaluatorResult, rich_print: bool = True):
    answer = evaluation.strigify_answer()
    if rich_print:
        rich.print(f"[bold blue]🔎 Query ID[/bold blue]: {evaluation.qid}")
        rich.print(f"[bold blue]👤 Agent[/bold blue]: {evaluation.agent}")
        rich.print(f"[bold blue]Parsed Answer[/bold blue]: {answer}")
        rich.print("")
    else:
        print(f"Query ID: {evaluation.qid}")
        print(f"Agent: {evaluation.agent}")
        print(f"Parsed Answer: {answer}")
        print("")


def _render_pairwise_game_evaluation(evaluation: PairwiseGameEvaluatorResult, rich_print: bool = True):
    answer = evaluation.strigify_answer()
    if rich_print:
        rich.print(f"[bold blue]🔎 Query ID[/bold blue]: {evaluation.qid}")
        rich.print(f"[bold blue]👤 Agent A[/bold blue]: {evaluation.agent_a}")
        rich.print(f"[bold blue]👤 Agent B[/bold blue]: {evaluation.agent_b}")
        rich.print(f"[bold blue]Parsed Answer[/bold blue]: {answer}")
        rich.print("")
    else:
        print(f"Query ID: {evaluation.qid}")
        print(f"Agent A: {evaluation.agent_a}")
        print(f"Agent B: {evaluation.agent_b}")
        print(f"Parsed Answer: {answer}")
        print("")


def _render_retrieval_evaluation(evaluation: RetrievalEvaluatorResult, rich_print: bool = True):
    answer = evaluation.strigify_answer()
    if rich_print:
        rich.print(f"[bold blue]🔎 Query ID[/bold blue]: {evaluation.qid}")
        rich.print(f"[bold blue]📜 Document ID[/bold blue]: {evaluation.did}")
        rich.print(f"[bold blue]Parsed Answer[/bold blue]: {answer}")
        rich.print("")
    else:
        print(f"Query ID: {evaluation.qid}")
        print(f"Document ID: {evaluation.did}")
        print(f"Parsed Answer: {answer}")
        print("")


def render_evaluation(
    evaluation: RetrievalEvaluatorResult | AnswerEvaluatorResult | PairwiseGameEvaluatorResult | EloTournamentResult,
    rich_print: bool = True,
):
    if isinstance(evaluation, EloTournamentResult):
        return _render_elo_tournament(evaluation, rich_print)
    if isinstance(evaluation, RetrievalEvaluatorResult) and evaluation.score is None and not evaluation.reasoning:
        return

    if isinstance(evaluation, RetrievalEvaluatorResult):
        return _render_retrieval_evaluation(evaluation, rich_print)
    if isinstance(evaluation, AnswerEvaluatorResult):
        return _render_answer_evaluation(evaluation, rich_print)
    if isinstance(evaluation, PairwiseGameEvaluatorResult):
        return _render_pairwise_game_evaluation(evaluation, rich_print)


def render_retrieval_summary(
    results: Dict[str, Dict[str, float]],
    metrics: list[str],
    relevance_threshold: int = 0,
    rich_print_enabled: bool = True,
):
    if not results:
        return
    key_metric = metrics[0]
    max_agent_len = max([len(agent) for agent in results.keys()]) + 3
    max_metric_len = max([len(metric) for metric in metrics])
    sorted_agents = sorted(results.items(), key=lambda x: x[1][key_metric], reverse=True)
    if rich_print_enabled:
        rich.print("---[bold cyan] Retrieval Scores [/bold cyan] ---")
        if relevance_threshold > 0:
            rich.print(f"[bold yellow]Relevance threshold: {relevance_threshold}[/bold yellow]")
        header = f"[bold magenta]{'Agent Name':<{max_agent_len}}"
        header += "\t".join([f"{m:<{max_metric_len}}" for m in metrics])
        header += "[/bold magenta]"
        rich.print(f"[bold cyan]{header}[/bold cyan]")
        for agent, scores in sorted_agents:
            row = f"[bold white]{agent:<{max_agent_len}}[/bold white]"
            row += "\t".join([f"{scores[metric]:<{max_metric_len},.4f}" for metric in metrics])
            rich.print(row)
        return
    # Plain print
    print(results)


def render_failed_evaluations(total_evaluations: int, failed_evaluations: int, rich_print: bool = True):
    if rich_print:
        rich.print("✅ Done!")
        if failed_evaluations > 0:
            rich.print(f"[bold red]Failed evaluations: {failed_evaluations}[/bold red]")
        rich.print(f"[bold green]Total evaluations: {total_evaluations}[/bold green]")
        return
    print("✅ Done!")
    if failed_evaluations > 0:
        print(f"Failed evaluations: {failed_evaluations}")
    print(f"Total evaluations: {total_evaluations}")
