"""Evaluator with a domain expert persona"""


from tenacity import RetryError

from ragelo.doc_evaluators.base_doc_evaluator import (
    DocumentEvaluator,
    DocumentEvaluatorFactory,
)
from ragelo.logger import logger


@DocumentEvaluatorFactory.register("domain_expert")
class DomainExpertEvaluator(DocumentEvaluator):
    sys_prompt = """You are a domain expert in {domain_long}.{company_prompt_1} You are tasked with evaluating the performance of a retrieval system for question answering in this domain. The question answering system will be used by internal users {company_prompt_2}{domain_short}. They are interested in a retrieval system that provides relevant passages based on their questions."""

    reason_prompt = """Query: {query}
Document passage: {doc_content}
Please think in steps about the relevance of the retrieved document given the original query. Consider the query, the document title, and the document passage. Reason whether the document is not relevant to the query, somewhat relevant to the query, or highly relevant to the query.
Use the following guidelines to reason about the relevance of the retrieved document:
- Not Relevant:
    The document contains information that is unrelated, outdated, or completely irrelevant to the query.
    The document may contain some keywords or phrases from the query, but the context and overall meaning do not align with the query's intent.
    The document may be from a different field or time period, rendering it irrelevant to the current query.
- Somewhat Relevant:
    The document contains some relevant information but lacks comprehensive details or context.
    The document may discuss a related topic or concept but not directly address the query.
    The information in the document is tangentially related to the query, but the primary focus remains different.
- Highly Relevant:
    The document directly addresses the main points of the query and provides comprehensive and accurate information.
    The document may cite relevant information directly applicable to the query.
    The document may be recent and from the same field as the query, enhancing its relevance.
General Guidelines:
    - Context Matters: Annotators should evaluate the relevance of documents within the specific{domain_short} context provided by the query. Understanding the nuances and domain-specific terminology is essential.
    - Content Overlap: Consider the extent of content overlap between the document and the query. Assess whether the document covers the core aspects of the query or only peripheral topics.
    - Neutrality: Base judgments solely on the content's relevance and avoid any personal opinions or biases.
    - Uncertainty: If uncertain about a relevance judgement, annotators default to a lower relevance.
    {extra_guidelines}"""

    score_prompt = """Given the previous reasoning, please assign a score of 0, 1, or 2 to the retrieved document given the particular query. The score meaning is as follows:
- 0: indicates that the retrieved document is not relevant to the query
- 1: the document is somewhat relevant to the query
- 2: the document is highly relevant to the query
Please only answer with a single number."""

    COMPANY_PROMPT_1 = "You work for {company}."
    COMPANY_PROMPT_2 = "of {company}"
    DOMAIN_SHORT = "but it also serves some of your external users like {domain_short}"

    def __init__(
        self,
        domain_long: str,
        company: str | None = None,
        domain_short: str | None = None,
        extra_guidelines: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.domain_long = domain_long
        self.company = company
        self.domain_short = f" {domain_short}" if domain_short else ""
        self.extra_guidelines = extra_guidelines
        self.sys_prompt = self.sys_prompt.format(
            domain_long=domain_long,
            company_prompt_1=self.COMPANY_PROMPT_1.format(company=company)
            if company
            else "",
            company_prompt_2=self.COMPANY_PROMPT_2.format(company=company)
            if company
            else "",
            domain_short=self.DOMAIN_SHORT.format(domain_short=domain_short)
            if domain_short
            else "",
        )
        self.extra_guidelines = extra_guidelines if extra_guidelines else ""
        self.reasoning_file = self.output_file.replace(".csv", "_reasoning.csv")

    def _build_reason_message(self, qid: str, did: str) -> str:
        query = self.queries[qid]
        document = self.documents[qid][did]
        reason_prompt = self.reason_prompt.format(
            query=query,
            doc_content=document,
            domain_short=self.domain_short if self.domain_short else "",
            extra_guidelines=self.extra_guidelines if self.extra_guidelines else "",
        )
        return reason_prompt

    def _process_second_answer(self, answer: str) -> str:
        return answer

    def _process_single_answer(self, qid, did) -> str:
        """Processes a single pair of qid, did in a two-shot manner"""
        reason_message = self._build_reason_message(qid, did)
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": reason_message},
        ]
        try:
            answer = self.openai_client(messages)
        except RetryError as e:
            logger.warning(f"Failed to fetch reasoning for document {qid} {did}")
            raise e
        except ValueError as e:
            logger.warning(f"Failed to parse reasoning for document {qid} {did}")
            raise e
        self._print_response(qid, did, answer=f"Reasoning: {answer}")
        self._dump_response(qid, did, answer, self.reasoning_file)

        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": self.score_prompt})
        try:
            answer = self.openai_client(messages)
            answer = self._process_second_answer(answer)
        except RetryError as e:
            logger.warning(f"Failed to fetch evaluation for document {qid} {did}")
            raise e
        except ValueError as e:
            logger.warning(f"Failed to parse evaluation for document {qid} {did}")
            raise e
        return answer
