from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator, RetrievalEvaluatorFactory
from ragelo.types.configurations import ReasonerEvaluatorConfig
from ragelo.types.types import RetrievalEvaluatorTypes
from ragelo.utils import string_to_template


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.REASONER)
class ReasonerEvaluator(BaseRetrievalEvaluator):
    """
    A document Evaluator that explicitly asks for the reasoning for why a document is relevant.
    """

    config: ReasonerEvaluatorConfig
    system_prompt = string_to_template("""
        You are an impartial expert document annotator, tasked with evaluating if a document contains relevant information to answer a question submitted by a user. 
        Your goal is to evaluate the relevancy of the documents given a user question, and write a concise reasoning for your decision.
            
        You should write one sentence reasoning wether the document is relevant or not for the user question. A document can be:
            - Not relevant: The document is not on topic.
            - Somewhat relevant: The document is on topic but does not fully answer the user question.
            - Very relevant: The document is on topic and answers the user question.
        """)

    user_prompt = string_to_template("""
        [user question]
        {{ query.query }}

        [document content]
        {{ document.text }}
        """)
