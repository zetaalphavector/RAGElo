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
ASPECTS = {
    "intent_match": "the match of the document to the query intent",
    "trustworthiness": "the trustworthiness of the document",
}


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.RDNAM)
class RDNAMEvaluator(BaseRetrievalEvaluator[RDNAMEvaluatorConfig]):
    config: RDNAMEvaluatorConfig
    result_type = RDNAMEvaluatorResult
    answer_format = RDNAMEvaluationAnswer
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

    def __init__(self, config: RDNAMEvaluatorConfig, llm_provider: BaseLLMProvider):
        super().__init__(config, llm_provider)
        self._response_schema = self._build_response_schema()

    @property
    def aspects(self) -> dict[str, str]:
        """What is scored before the overall relevance, by the name it is stored under."""
        return ASPECTS if self.config.use_aspects else {}

    def _build_response_schema(self) -> type[BaseModel]:
        fields: dict[str, tuple[type, FieldInfo]] = {
            "reasoning": (str, Field(description="A concise explanation of the scores."))
        }
        for name, aspect in self.aspects.items():
            description = f"An integer from 0 to {self.max_score} representing {aspect}."
            fields[name] = (int, Field(description=description, ge=0, le=self.max_score))
        fields["score"] = (int, self._score_field())
        judgment = create_model("RDNAMJudgment", **fields)  # type: ignore[call-overload]
        if not self.config.use_multiple_annotators:
            return judgment
        annotators = {f"annotator_{i}": (judgment, ...) for i in range(1, N_ANNOTATORS + 1)}
        return create_model("RDNAMAnnotators", **annotators)  # type: ignore[call-overload]

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
        """The annotators all answer in one call, and the stored judgment is their average."""
        response = llm_response.parsed_answer.model_dump()
        judgments = list(response.values()) if self.config.use_multiple_annotators else [response]
        answer = RDNAMEvaluationAnswer(
            reasoning="\n\n".join(judgment["reasoning"] for judgment in judgments),
            score=fmean(judgment["score"] for judgment in judgments),
            **{name: fmean(judgment[name] for judgment in judgments) for name in self.aspects},
        )
        return LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=answer)
