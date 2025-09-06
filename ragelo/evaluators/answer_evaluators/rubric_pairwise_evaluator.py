from typing import Type

from pydantic import BaseModel, Field, create_model

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import AnswerEvaluatorFactory
from ragelo.evaluators.answer_evaluators.pairwise_evaluator import PairwiseAnswerEvaluator
from ragelo.types.answer_formats import Criterion, CriterionEvaluation, RubricPairwiseAnswerAnswerFormat
from ragelo.types.configurations import RubricPairwiseEvaluatorConfig
from ragelo.types.evaluables import Document, PairwiseGame
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.query import Query
from ragelo.types.types import AnswerEvaluatorTypes
from ragelo.utils import call_async_fn, string_to_template


class RubricSchema(BaseModel):
    criteria: list[Criterion] = Field(description="The criteria to be used to evaluate the quality of the responses.")


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.RUBRIC_PAIRWISE)
class RubricPairwiseEvaluator(PairwiseAnswerEvaluator):
    config: RubricPairwiseEvaluatorConfig
    criteria_prompt = string_to_template("""
        You are a domain expert in {{ expert_in }}.{% if company %} You work for {{ company }}.{% endif %}
        Your task is to, given a user question and a set of relevant retrieved documents, create a rubric for evaluating the quality of reports generated written by other agents.
        You should think deeply and carefully about what questions should a complete and high-quality report that answer the question should answer. 
        Each criterion should be a short yes/no question that can be used to evaluate the quality of the responses. You should write 5 criteria.
        If a criterion is supported by a document, you should include the document ID in the supporting_documents list.
        """)

    criteria_user_prompt = string_to_template("""
        [User Question]
        {{ query.query }}

        [Retrieved Documents]
        {% for doc in documents %}
        [[{{doc.did}}]] {{doc.text}}
        --------------------------------
        {% endfor %}
        """)

    system_prompt = string_to_template("""
        You are a domain expert in {{ expert_in }}.{% if company %} You work for {{ company }}.{% endif %} 
        You are tasked with evaluating the quality of the responses provided by two AI assistants that were tasked with answering a user's question based on a set of documents retrieved by a search engine.
        When available, answers will cite specific documents by placing their IDs into square brackets.

        You will be provided with a list of criteria to evaluate the quality of the responses. 
        For each criterion, you should think carefully about which of two answers better answers the criterion, and assign, for each criterion, one of the following outputs:

        - A if Answer A clearly answers the criterion better than answer B
        - B if Answer B clearly answers the criterion better than answer A
        - C if Answer A and answer B are equally good and answer the criterion equally well
        - D if Answer A and answer B are equally bad and neither answers the criterion

        You should think carefully about the criteria and the answers, and assign the winner accordingly.

        ## Criteria
        {% for criteria in criteria.criteria %}
        Criterion: {{criteria.criterion_name}}
        Supporting Documents: {{criteria.supporting_documents}}
        Short Question: {{criteria.short_question}}
        --------------------------------
        {% endfor %}
        """)

    user_prompt = string_to_template("""
        [User Question]
            {{query.query}}

        [The Start of Assistant A's Answer]
            {{ game.agent_a_answer.text }}
        [The End of Assistant A's Answer]

        [The Start of Assistant B's Answer]
            {{ game.agent_b_answer.text }}
        [The End of Assistant B's Answer]""")

    criteria_cache: dict[str, Criterion] = {}
    answer_schema_cache: dict[str, Type[BaseModel]] = {}

    async def _build_criteria(self, query: Query, documents: list[Document]) -> RubricSchema:
        documents = self._filter_documents(query)
        context = {
            "expert_in": self.config.expert_in,
            "company": self.config.company,
            "documents": documents,
            "query": query,
        }
        criteria_prompt = self.criteria_prompt.render(context)
        user_prompt = self.criteria_user_prompt.render(context)
        llm_input = LLMInputPrompt(system_prompt=criteria_prompt, user_message=user_prompt)
        llm_response = await self.llm_provider.call_async(llm_input, response_schema=RubricSchema)
        criteria_models = {}
        for criteria in llm_response.parsed_answer.criteria:
            criteria_models[criteria.criterion_name] = create_model(
                criteria.criterion_name,
                reasoning=(
                    str,
                    Field(description="A brief explanation about your judgement, and why you chose the winner"),
                ),
                winner=(
                    str,
                    Field(description="The winner of the criterion"),
                ),
            )

        evaluation_schema = create_model("EvaluationSchema", **criteria_models)
        self.answer_schema_cache[query.qid] = evaluation_schema
        return llm_response.parsed_answer

    def _build_message_pairwise(self, query: Query, game: PairwiseGame) -> LLMInputPrompt:
        # Check if this query already have a criteria set. Otherwise, get it first.
        if query.qid not in self.criteria_cache:
            self.criteria_cache[query.qid] = call_async_fn(
                self._build_criteria, query, list(query.retrieved_docs.values())
            )

        criteria = self.criteria_cache[query.qid]
        self.config.llm_response_schema = self.answer_schema_cache[query.qid]
        system_prompt = self.system_prompt.render(
            expert_in=self.config.expert_in, criteria=criteria, company=self.config.company
        )
        user_prompt = self.user_prompt.render(
            query=query,
            game=game,
        )
        return LLMInputPrompt(system_prompt=system_prompt, user_message=user_prompt)

    def _process_answer(self, llm_response: LLMResponseType, query: Query | None = None) -> LLMResponseType:
        response_dict = llm_response.parsed_answer.model_dump()
        agent_a_wins = 0
        agent_b_wins = 0
        equally_good = 0
        equally_bad = 0
        criteria: list[CriterionEvaluation] = []

        for response in response_dict.values():
            criterion = CriterionEvaluation(
                criterion=self.criteria_cache[query.qid], reasoning=response["reasoning"], winner=response["winner"]
            )
            criteria.append(criterion)
            if response["winner"] == "A":
                agent_a_wins += 1
            elif response["winner"] == "B":
                agent_b_wins += 1
            elif response["winner"] == "C":
                equally_good += 1
            else:
                equally_bad += 1

        parsed_answer = RubricPairwiseAnswerAnswerFormat(
            criteria=criteria,
            agent_a_wins=agent_a_wins,
            agent_b_wins=agent_b_wins,
            equally_good=equally_good,
            equally_bad=equally_bad,
            winner="A" if agent_a_wins > agent_b_wins else "B" if agent_a_wins < agent_b_wins else "C",
        )

        return LLMResponseType(
            raw_answer=llm_response.raw_answer,
            parsed_answer=parsed_answer,
        )
