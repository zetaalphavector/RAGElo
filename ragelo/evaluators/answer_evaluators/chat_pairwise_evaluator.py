"""Answer Evaluator for conversations between two agents."""

from __future__ import annotations

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
)
from ragelo.evaluators.answer_evaluators.pairwise_evaluator import (
    PairwiseAnswerEvaluator,
)
from ragelo.types.configurations import PairwiseEvaluatorConfig
from ragelo.types.evaluables import PairwiseGame
from ragelo.types.query import Query
from ragelo.types.types import AnswerEvaluatorTypes


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.CHAT_PAIRWISE)
class ChatPairwiseEvaluator(PairwiseAnswerEvaluator):
    config: PairwiseEvaluatorConfig
    prompt = """
Please act as an impartial judge and evaluate the quality of the responses provided \
by two AI assistants, tasked to provide answers grounded in a set of documents \
retrieved by a search engine in order to satisfy the user's intent displayed below.
You should choose the assistant that best satisfies the user's intent based on a set \
of reference documents that may or may not be relevant. {citations}
{document_rel}
Your evaluation should consider the evaluation objectives listed below.
If an answer contains non-relevant details, it should not be preferred over one that \
only use relevant information.
Begin your evaluation by examining each agent separately and explaining if each answer \
is useful towards satisfying the user's intent at each iteration. Then, provide a \
short explanation how the agent performed overall based on the evaluation objectives. \
Finally, compare the conversations of the two agents and provide a short explanation \
on their differences. Avoid any position biases and ensure that the order in which \
the conversations were presented does not influence your decision. Do not allow the \
length of the responses to influence your evaluation. Be as objective as possible.
After providing your explanation, output your final verdict by strictly following \
this format: 'A' if assistant A is better, 'B' if assistant B is better, \
and 'C' for a tie.

[User Intent]
{query}

[Evaluation Objectives]
{factors}

[Reference Documents]
{documents}

[The Start of Conversation with Assistant A]
{answer_a}
[The End of Conversation with Assistant A]

[The Start of Conversation with Assistant B]
{answer_b}
[The End of Conversation with Assistant B]
""".strip()

    def _build_message_pairwise(self, query: Query, game: PairwiseGame) -> str | list[dict[str, str]]:
        documents = self._prepare_documents(query)
        query_metadata = self._get_usable_fields_from_metadata(
            self.prompt, query.metadata, skip_fields=[self.config.query_placeholder]
        )
        answer_a_metadata = self._get_usable_fields_from_metadata(
            self.prompt,
            game.agent_a_answer.metadata,
            skip_fields=[self.config.answer_placeholder],
        )
        answer_b_metadata = self._get_usable_fields_from_metadata(
            self.prompt,
            game.agent_b_answer.metadata,
            skip_fields=[self.config.answer_placeholder],
        )
        if self.config.has_citations:
            citations = self.citations_prompt
        else:
            citations = ""

        if not game.agent_a_answer.conversation or not game.agent_b_answer.conversation:
            raise ValueError("The conversation of the agents cannot be empty for the chat_pairwise evaluator")
        conversation_a = "\n".join([str(msg) for msg in game.agent_a_answer.conversation])
        conversation_b = "\n".join([str(msg) for msg in game.agent_b_answer.conversation])
        formatters = {
            self.config.query_placeholder: query.query,
            self.config.documents_placeholder: documents,
            "answer_a": conversation_a,
            "answer_b": conversation_b,
            "citations": citations,
            "factors": self.factors,
            "document_rel": self.documents_prompt,
            **query_metadata,
            **answer_a_metadata,
            **answer_b_metadata,
        }
        return self.prompt.format(**formatters)
