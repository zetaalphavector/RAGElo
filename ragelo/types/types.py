from dataclasses import dataclass


@dataclass
class Query:
    qid: str
    query: str


@dataclass
class Document:
    query: Query
    did: str
    text: str


@dataclass
class AgentAnswer:
    query: Query
    agent: str
    text: str
