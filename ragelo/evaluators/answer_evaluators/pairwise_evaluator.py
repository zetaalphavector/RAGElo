from ragelo.evaluators.answer_evaluators.base_answer_evaluator import AnswerEvaluatorFactory, BaseAnswerEvaluator
from ragelo.types.configurations import PairwiseEvaluatorConfig
from ragelo.types.evaluables import PairwiseGame
from ragelo.types.formats import LLMInputPrompt
from ragelo.types.query import Query
from ragelo.types.types import AnswerEvaluatorTypes
from ragelo.utils import string_to_template


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.PAIRWISE)
class PairwiseAnswerEvaluator(BaseAnswerEvaluator):
    """An evaluator that evaluates RAG-based answers pairwise, with document reasoning and citations."""

    config: PairwiseEvaluatorConfig
    user_prompt_document = "[{did}] {doc}"
    user_prompt_annotation = "[{did}] {annotation}"
    user_prompt_document_and_annotation = "[{{ d.did }}] Content: {{ d.text }}\n Evaluation: {{ d.evaluation.answer }}"

    system_prompt = string_to_template("""
        Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the question of a user, based on a set of documents retrieved by a search engine that may or may not be relevant to the question. 
        When available, answers will cite specific documents by placing their IDs into square brackets.
        {%- if doc and (annotation or reasoning) %}
        You will be provided with the text of each reference document and its relevance evaluation.
        {%- elif doc %}
        You will be provided with the text of each reference document.
        {%- elif annotation or reasoning %}
        For each cited document, you will be provided with its relevance evaluation.
        {%- endif %}
        {%- if not reasoning %}
        Each document is scored in a scale of 0 to 2, where:
            - 0: the document is not relevant to the query
            - 1: the document is somewhat relevant to the query
            - 2: the document is highly relevant to the query
        {%- endif %}

        You should choose the assistant that best answers the user's question.

        ## Evaluation Guidelines
        - Your evaluation should consider factors such as {{ factors }}.
        - Details are only useful if they answer the user question.
        - If an answer contains non-relevant details, it should not be preferred over one that only use relevant information.
        - Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
        - Do not allow the length of the responses to influence your evaluation.
        - Be as objective as possible.

        ## Workflow
        First, you should analyze each of the two answers, explaining whether or not each of them correctly answers the user's question, based on the relevant documents retrieved. 
        Then, you should compare the two responses and provide a short explanation on their differences, explaining in which aspects each answer is better or worst than the other. 
        After providing your explanation, output your final verdict by strictly following his format: "A" if assistant A is better, "B" if assistant B is better, or "C" for a tie.""")

    user_prompt = string_to_template("""
        [User Question]
        {{ query.query }}
        {%- if documents %}
        [Reference Documents]
        {%- for d in documents %}
        {%- if doc and (annotation or reasoning) %}
            Document ID: [{{ d.did }}]
            Content: {{ d.text }}
            Relevance: {% if reasoning %} {{ d.evaluation.answer.reasoning }} {% else %} {{ d.evaluation.answer.score }} {% endif %}
        ------------------
        {%- elif doc %}
            [{{ d.did }}]: {{ d.text }}
        {%- elif annotation or reasoning %}
            [{{d.did }}] {% if reasoning %} {{ d.evaluation.answer.reasoning }} {% else %} {{ d.evaluation.answer.score }} {% endif %}"
        {% endif -%}
        {% endfor %}
        {% endif -%}

        [The Start of Assistant A's Answer]
            {{ game.agent_a_answer.text }}
        [The End of Assistant A's Answer]

        [The Start of Assistant B's Answer]
            {{ game.agent_b_answer.text }}
        [The End of Assistant B's Answer]""")

    def _build_message_pairwise(self, query: Query, game: PairwiseGame) -> LLMInputPrompt:
        documents = self._filter_documents(query)
        context = {
            "factors": self.config.factors,
            "query": query,
            "documents": documents,
            "game": game,
            "doc": self.config.include_raw_documents,
            "annotation": self.config.include_relevance_score,
            "reasoning": self.config.include_relevance_reasoning,
        }
        return LLMInputPrompt(
            system_message=self.system_prompt.render(**context),
            user_message=self.user_prompt.render(**context),
        )
