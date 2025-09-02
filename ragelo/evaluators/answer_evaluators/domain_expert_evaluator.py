"""Answer Evaluator with a domain expert persona"""

from textwrap import dedent

from jinja2 import Template

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
)
from ragelo.evaluators.answer_evaluators.pairwise_evaluator import (
    PairwiseAnswerEvaluator,
)
from ragelo.types.configurations import PairwiseDomainExpertEvaluatorConfig
from ragelo.types.evaluables import PairwiseGame
from ragelo.types.formats import LLMInputPrompt
from ragelo.types.query import Query
from ragelo.types.types import AnswerEvaluatorTypes


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.DOMAIN_EXPERT)
class PairwiseDomainExpertEvaluator(PairwiseAnswerEvaluator):
    config: PairwiseDomainExpertEvaluatorConfig
    system_template = Template(
        dedent(
            """
            You are a domain expert in {{ expert_in }}.{% if company %} You work for {{ company }}.{% endif %} You are tasked with evaluating the quality of the responses provided by two AI assistants that were tasked with answering a user's question based on a set of documents retrieved by a search engine.
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
            Your evaluation should consider factors such as:
            {%- for g in factors %}
                - {{ g }}
            {% endfor %}
            - Details are only useful if they answer the user question.
            - If an answer contains non-relevant details, it should not be preferred over one that only use relevant information.
            - Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
            - Do not allow the length of the responses to influence your evaluation.
            - Be as objective as possible.
            - Remember that you are in expert in {{ expert_in }}. Make your judgement accordingly.

            ## Workflow
            First, you should analyze each of the two answers, explaining whether or not each of them correctly answers the user's question, based on the relevant documents retrieved and your expertise.
            Then, you should compare the two responses and provide a short explanation on their differences, explaining in which aspects each answer is better or worst than the other. 
            After providing your explanation, output your final verdict by strictly following his format: "A" if assistant A is better, "B" if assistant B is better, or "C" for a tie.
            """
        )
    )
    user_prompt = Template(
        dedent(
            """
            [User Question]
            {{ query.query }}
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

            [The Start of Assistant A's Answer]
                {{ game.agent_a_answer.text }}
            [The End of Assistant A's Answer]

            [The Start of Assistant B's Answer]
                {{ game.agent_b_answer.text }}
            [The End of Assistant B's Answer]
            """
        )
    )

    def _build_message_pairwise(self, query: Query, game: PairwiseGame) -> LLMInputPrompt:
        documents = self._filter_documents(query)
        context = {
            "factors": self.config.factors,
            "query": query.query,
            "documents": documents,
            "game": game,
            "doc": self.config.include_raw_documents,
            "annotation": self.config.include_annotations,
            "expert_in": self.config.expert_in,
            "company": self.config.company,
        }

        return LLMInputPrompt(
            system_message=self.system_prompt.render(**context),
            user_message=self.user_prompt.render(**context),
        )
