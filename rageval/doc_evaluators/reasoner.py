import csv
import os

from tenacity import RetryError

from rageval.doc_evaluators import DocumentEvaluator, DocumentEvaluatorFactory
from rageval.logger import logger


@DocumentEvaluatorFactory.register("reasoner")
class ReasonerDocEval(DocumentEvaluator):
    """
    A document Evaluator that only outputs the reasoning for why a document
    is relevant.
    """

    def __init__(
        self,
        query_path: str,
        documents_path: str,
        output_file: str,
        prompt_name: str,
        model_name: str = "gpt-4",
        credentials_file: str | None = None,
        verbose: bool = False,
        force: bool = False,
    ):
        super().__init__(
            query_path,
            documents_path,
            output_file,
            "reasoner",
            model_name,
            credentials_file,
            verbose,
            force,
        )

    def get_answers(self):
        max = 10
        skip_docs = set()
        if os.path.isfile(self.output_file) and not self.force:
            for line in csv.reader(open(self.output_file)):
                qid, did, answer = line
                skip_docs.add((qid, did))
        if len(skip_docs) > 0:
            logger.info(f"Skipping {len(skip_docs)} documents")

        if self.verbose:
            try:
                from rich.progress import track

                q_iterator = track(self.queries, description="Annotating Documents")
            except ImportError:
                q_iterator = self.queries
        q_iterator = self.queries
        for qid in q_iterator:
            if max == 0:
                break
            max -= 1
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
                logger.debug(answer)
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
                        writer.writerow(["qid", "did", "answer"])

                with open(self.output_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([qid, did, answer])

    def __build_message(self, document: str, query: str) -> str:
        return self.prompt.format(user_question=query, doc_content=document)
