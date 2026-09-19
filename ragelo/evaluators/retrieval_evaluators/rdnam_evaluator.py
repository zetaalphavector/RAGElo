"""Evaluator based on the paper: Paul Thomas, Seth Spielman, Nick Craswell and
Bhaskar Mitra. Large language models can accurately predict searcher preferences.
https://arxiv.org/abs/2309.10621
"""

from statistics import fmean

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.answer_formats import RDNAMEvaluationAnswer
from ragelo.types.configurations import RDNAMEvaluatorConfig
from ragelo.types.evaluables import Document
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.query import Query
from ragelo.types.results import RDNAMEvaluatorResult
from ragelo.types.types import RetrievalEvaluatorTypes
from ragelo.utils import string_to_template

N_ANNOTATORS = 5


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.RDNAM)
class RDNAMEvaluator(BaseRetrievalEvaluator[RDNAMEvaluatorConfig]):
    config: RDNAMEvaluatorConfig
    relevance_grades = (
        "not relevant, should never be shown for this query",
        "relevant, may be partly helpful but might contain other irrelevant content",
        "highly relevant, very helpful for this query",
    )
    system_prompt = string_to_template("""
        {% if annotator_role %}{{ annotator_role }} {% endif %} Given a query and a document, you must provide a score on an integer scale of 0 to {{ max_score }} with the following meanings:
        {%- for grade in relevance_grades | reverse %}
        {{ loop.revindex0 }} = {{ grade }}
        {%- endfor %}
        {%- if not custom_grades %}
        Assume that you are writing a report on the subject of the topic. If you would use any of the information contained in the document in such a report, mark it 1. If the document is primarily about the topic, or contains vital information about the topic, mark it 2. Otherwise, mark it 0.
        {%- endif %}
        """)
    user_prompt = string_to_template("""
        # Query
        A person has typed {{ query.query }} into a search engine.
        {% if query.metadata and (query.metadata.description or query.metadata.narrative) %}
        They were looking for: {{ query.metadata.description }}
        {{ query.metadata.narrative }}
        {% endif %}

        # Result
        Consider the following document.
        ---BEGIN DOCUMENT CONTENT---
        {{ document.text }}
        ---END DOCUMENT CONTENT---

        # Instructions
        Split this problem into steps:
        Consider the underlying intent of the search.
        {%- if use_aspects %}
        Measure how well the content matches a likely intent of the query.
        Measure how trustworthy the web page is.
        {%- endif %}
        Consider the aspects above and relative importance of each, and decide on a final overall score.
        {%- if multiple %}
        We asked five search engine raters to evaluate the relevance of the web page for the query.
        Each rater used their own independent judgement.
        {%- endif %}""")
    result_type = RDNAMEvaluatorResult
    answer_format = RDNAMEvaluationAnswer

    def __init__(self, config: RDNAMEvaluatorConfig, llm_provider: BaseLLMProvider):
        super().__init__(config, llm_provider)
        self._response_schema = self._build_response_schema()

    def _build_response_schema(self) -> type[BaseModel]:
        fields: dict[str, tuple[type, FieldInfo]] = {"score": (int, self._score_field())}
        if self.config.use_aspects:
            fields["intent_match"] = (int, self._aspect_field("the match of the document to the query intent"))
            fields["trustworthiness"] = (int, self._aspect_field("the trustworthiness of the document"))
        judgment = create_model("RDNAMJudgment", **fields)  # type: ignore[call-overload]
        if not self.config.use_multiple_annotators:
            return judgment
        annotators = {f"annotator_{i}": (judgment, ...) for i in range(1, N_ANNOTATORS + 1)}
        return create_model("RDNAMAnnotators", **annotators)  # type: ignore[call-overload]

    def _aspect_field(self, aspect: str) -> FieldInfo:
        return Field(
            description=f"An integer from 0 to {self.max_score} representing {aspect}.", ge=0, le=self.max_score
        )

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        context = self._prompt_context(query, document) | {
            "annotator_role": self.config.annotator_role,
            "use_aspects": self.config.use_aspects,
            "multiple": self.config.use_multiple_annotators,
        }
        return LLMInputPrompt(
            system_prompt=self.system_prompt.render(**context),
            user_message=self.user_prompt.render(**context),
            llm_response_schema=self._response_schema,
        )

    def _process_answer(self, llm_response: LLMResponseType[BaseModel], query: Query) -> LLMResponseType[BaseModel]:
        parsed = llm_response.parsed_answer
        if parsed is None:
            return llm_response
        response = parsed.model_dump()
        judgments = list(response.values()) if self.config.use_multiple_annotators else [response]
        answer = RDNAMEvaluationAnswer(score=fmean(judgment["score"] for judgment in judgments))
        if self.config.use_aspects:
            answer.intent_match = fmean(judgment["intent_match"] for judgment in judgments)
            answer.trustworthiness = fmean(judgment["trustworthiness"] for judgment in judgments)
        return LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=answer)
