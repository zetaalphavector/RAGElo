from ragelo.doc_evaluators import DocumentEvaluator, DocumentEvaluatorFactory


@DocumentEvaluatorFactory.register("reasoner")
class ReasonerEvaluator(DocumentEvaluator):
    """
    A document Evaluator that only outputs the reasoning for why a document
    is relevant.
    """

    prompt = """You are an expert document annotator, evaluating if a document contains relevant information to anser a question submitted by a user. Please act as an impartial relevance annotator for a search engine. Your goal is to evaluate the relevancy of the documents given a user question.
    You should write one sentence explaining why the document is relevant or not for the user question. A document can be:
    - Not relevant: The document is not on topic.
    - Somewhat relevant: The document is on topic but does not fully answer the user question.
    - Very relevant: The document is on topic and answers the user question.
    [user question]
    {user_question}

    [document content]
    {doc_content}"""  # noqa: E501

    def _build_message(self, qid: str, did: str) -> str:
        query = self.queries[qid]
        document = self.documents[qid][did]
        return self.prompt.format(user_question=query, doc_content=document)

    def _process_answer(self, answer: str) -> str:
        return answer
