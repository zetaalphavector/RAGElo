import csv
import os

from tenacity import RetryError

from rageval.doc_evaluators import DocumentEvaluator, DocumentEvaluatorFactory
from rageval.logger import logger


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

    def get_answers(self):
        print(self.verbose)
        skip_docs = set()
        if os.path.isfile(self.output_file) and not self.force:
            for line in csv.reader(open(self.output_file)):
                qid, did, answer = line
                skip_docs.add((qid, did))
        if self.force and os.path.isfile(self.output_file):
            # remove the file
            os.remove(self.output_file)
        if len(skip_docs) > 0:
            logger.info(f"Skipping {len(skip_docs)} documents")
        q_iterator = self.queries
        if self.verbose:
            try:
                from rich.progress import track

                q_iterator = track(self.queries, description="Annotating Documents")
            except ImportError:
                pass

        for qid in q_iterator:
            for did in self.documents[qid]:
                if (qid, did) in skip_docs:
                    logger.debug(f"Skipping {qid} {did}")
                    continue
                message = self.__build_message(
                    self.documents[qid][did], self.queries[qid]
                )
                try:
                    answer = self.openai_client(message)
                except RetryError:
                    logger.warning(f"Failed to annotate document {qid} {did}")
                    continue
                if self.verbose:
                    logger.info(
                        "[bold cyan]Query       [/bold cyan]: "
                        f"[not bold cyan]{self.queries[qid]}[/not bold cyan]"
                    )
                    logger.info(f"[bold cyan]Document ID [/bold cyan]: {did}")
                    logger.info(
                        "[bold cyan]Evaluation  [/bold cyan]: "
                        f"[not bold]{answer}[/not bold]"
                    )
                    logger.info("")
                if not os.path.isfile(self.output_file):
                    with open(self.output_file, "w") as f:
                        writer = csv.writer(f)
                        writer.writerow(["query_id", "did", "answer"])

                with open(self.output_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([qid, did, answer])

    def __build_message(self, document: str, query: str) -> str:
        return self.prompt.format(user_question=query, doc_content=document)
