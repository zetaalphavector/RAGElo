from dataclasses import dataclass


@dataclass
class Query:
    qid: str
    query: str


@dataclass
class Document:
    qid: str
    did: str
    text: str


@dataclass
class Answer:
    qid: str
    agent: str
    text: str
