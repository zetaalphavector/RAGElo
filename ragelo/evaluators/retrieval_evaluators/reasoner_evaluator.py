from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator, RetrievalEvaluatorFactory
from ragelo.types.configurations import ReasonerEvaluatorConfig
from ragelo.types.types import RetrievalEvaluatorTypes
from ragelo.utils import string_to_template


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.REASONER)
class ReasonerEvaluator(BaseRetrievalEvaluator[ReasonerEvaluatorConfig]):
    """
    A document Evaluator that explicitly asks for the reasoning for why a document is relevant.
    """

    config: ReasonerEvaluatorConfig
    relevance_grades = (
        (
            "Not relevant: The document contains no information that helps answer the user question, "
            "even if it is on the same topic or shares keywords with it."
        ),
        "Somewhat relevant: The document contains partial or indirect information that helps answer the user question.",
        "Very relevant: The document contains the answer to the user question, even if surrounded by other content.",
    )
    system_prompt = string_to_template("""
        You are an impartial expert document annotator, tasked with evaluating if a document contains relevant information to answer a question submitted by a user. 
        Your goal is to evaluate the relevancy of the documents given a user question, and write a concise reasoning for your decision.
            
        You should write one sentence reasoning wether the document is relevant or not for the user question. A document can be:
        {%- for grade in relevance_grades %}
            - {{ grade }}
        {%- endfor %}
        """)

    user_prompt = string_to_template("""
        [user question]
        {{ query.query }}

        [document content]
        {{ document.text }}
        """)
