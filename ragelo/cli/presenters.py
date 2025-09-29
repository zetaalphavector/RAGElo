from __future__ import annotations

import json
from typing import Any, Dict

import rich
from pydantic import BaseModel

from ragelo.types.results import AnswerEvaluatorResult, EloTournamentResult, RetrievalEvaluatorResult


def _stringify_answer(answer: Any) -> str:
    if isinstance(answer, dict):
        return json.dumps(answer, indent=4, ensure_ascii=False)
    if isinstance(answer, BaseModel):
        return answer.model_dump_json(indent=4)
    return str(answer)


def render_evaluation(
    evaluation: RetrievalEvaluatorResult | AnswerEvaluatorResult | EloTournamentResult,
    rich_print: bool = True,
):
    if isinstance(evaluation, EloTournamentResult):
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
        return
    if evaluation.answer is None:
        return
    answer = _stringify_answer(evaluation.answer)

    if isinstance(evaluation, RetrievalEvaluatorResult):
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
        return

    # AnswerEvaluatorResult
    assert isinstance(evaluation, AnswerEvaluatorResult)
    agent_a = getattr(evaluation, "agent_a", None)
    agent_b = getattr(evaluation, "agent_b", None)
    agent = getattr(evaluation, "agent", None)
    if rich_print:
        rich.print(f"[bold blue]🔎 Query ID[/bold blue]: {evaluation.qid}")
        if agent_a and agent_b:
            rich.print(f"[bold bright_cyan] {agent_a:<18} [/bold bright_cyan] 🆚  [bold red] {agent_b}[/bold red]")
        elif agent:
            rich.print(f"[bold bright_cyan]🕵️ Agent[/bold bright_cyan]: {agent}")
        rich.print(f"[bold blue]Parsed Answer[/bold blue]: {answer}")
        rich.print("")
    else:
        print(f"Query ID: {evaluation.qid}")
        if agent_a and agent_b:
            print(f"{agent_a} vs {agent_b}")
        elif agent:
            print(f"Agent: {agent}")
        print(f"Parsed Answer: {answer}")
        print("")


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
