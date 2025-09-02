"""Answer Evaluator for conversations between two agents."""

from textwrap import dedent

from jinja2 import Template

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
)
from ragelo.evaluators.answer_evaluators.pairwise_evaluator import (
    PairwiseAnswerEvaluator,
)
from ragelo.types.configurations import PairwiseEvaluatorConfig
from ragelo.types.evaluables import PairwiseGame
from ragelo.types.formats import LLMInputPrompt
from ragelo.types.query import Query
from ragelo.types.types import AnswerEvaluatorTypes


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.CHAT_PAIRWISE)
class ChatPairwiseEvaluator(PairwiseAnswerEvaluator):
    config: PairwiseEvaluatorConfig
    system_prompt = Template(
        dedent(
            """
                Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the question of a user, based on a set of documents retrieved by a search engine that may or may not be relevant to the question.
                When available, answers will cite specific documents by placing their IDs into square brackets.
                {%- if doc and annotation %}
                You will be provided with the text of each reference document and its relevance evaluation.
                {%- elif doc %}
                You will be provided with the text of each reference document.
                {%- elif annotation %}
                For each cited document, you will be provided with its relevance evaluation
                {%- endif %}
                You should choose the assistant that best answers the user's question.

                ## Evaluation Guidelines
                Your evaluation should consider the evaluation objectives listed below:
                - If an answer contains non-relevant details, it should not be preferred over one that only use relevant information.
                - Avoid any position biases and ensure that the order in which the responses were presented does not 
                - Do not allow the length of the responses to influence your evaluation.
                - Be as objective as possible.

                ## Workflow
                - Begin your evaluation by examining each agent separately and explaining if each answer is useful towards satisfying the user's intent at each iteration.
                - Then, provide a short explanation how the agent performed overall based on the evaluation objectives
                - Finally, compare the conversations of the two agents and provide a short explanation on their differences.
                - After providing your explanation, output your final verdict by strictly following this format: 'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie.
            """
        )
    )
    user_prompt = Template(
        dedent(
            """
            [User Intent]
            {{ query.query }}

            [Evaluation Objectives]
            {{ factors }}

            {%- if documents %}
            [Reference Documents]
            {%- for d in documents %}
            {%- if doc and annotation %}
                Document ID: [{{ d.did }}]
                Content: {{ d.text }}
                Relevance Evaluation: {{ d.evaluation.answer }}
            ------------------
            {%- elif doc %}
                [{{ d.did }}]: {{ d.text }}
            {%- elif annotation %}
                [{{d.did }}] {{ d.evaluation.answer }}"
            {% endif -%}
            {% endfor %}
            {% endif -%}

            [The Start of Conversation with Assistant A]
            {% for msg in game.agent_a_answer.conversation %}
            {{ msg }}
            {% endfor %}
            [The End of Conversation with Assistant A]

            [The Start of Conversation with Assistant B]
            {% for msg in game.agent_b_answer.conversation %}
            {{ msg }}
            {% endfor %}
            [The End of Conversation with Assistant B]
            """
        )
    )

    def _build_message_pairwise(self, query: Query, game: PairwiseGame) -> LLMInputPrompt:
        documents = self._filter_documents(query)

        if not game.agent_a_answer.conversation or not game.agent_b_answer.conversation:
            raise ValueError("The conversation of the agents cannot be empty for the chat_pairwise evaluator")

        context = {
            "factors": self.config.factors,
            "query": query.query,
            "documents": documents,
            "answer_a": game.agent_a_answer.text,
            "answer_b": game.agent_b_answer.text,
            "doc": self.config.include_raw_documents,
            "annotation": self.config.include_annotations,
        }
        return LLMInputPrompt(
            system_message=self.system_prompt.format(**context),
            user_message=self.user_prompt.format(**context),
        )
