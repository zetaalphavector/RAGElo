from dataclasses import dataclass, field


@dataclass
class Query:
    qid: str
    query: str


@dataclass
class Document:
    qid: str
    did: str
    text: str
