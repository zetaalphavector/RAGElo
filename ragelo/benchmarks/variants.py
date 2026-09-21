"""The retrieval evaluator configurations the retrieval datasets can be judged with, by `--evaluator` name."""

from __future__ import annotations

from typing import Any

# The definitions the NIST assessors labeled TREC DL 2023 with, from the LLMJudge challenge README.
TREC_DL_GRADES = [
    "Irrelevant: The passage has nothing to do with the query.",
    "Related: The passage seems related to the query but does not answer it.",
    (
        "Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, "
        "or hidden amongst extraneous information."
    ),
    "Perfectly relevant: The passage is dedicated to the query and contains the exact answer.",
]
# The reasoner's grades until the answer-focused wording replaced them, kept to reproduce that comparison.
TOPICAL_GRADES = [
    "Not relevant: The document is not on topic.",
    "Somewhat relevant: The document is on topic but does not fully answer the user question.",
    "Very relevant: The document is on topic and answers the user question.",
]
LOWER_WHEN_UNSURE = "If you are uncertain between two relevance grades, choose the lower one."

RETRIEVAL_VARIANTS: dict[str, dict[str, Any]] = {
    "reasoner": {"evaluator_name": "reasoner"},
    "rdnam": {"evaluator_name": "RDNAM"},
    "rdnam_aspects": {"evaluator_name": "RDNAM", "use_aspects": True},
    "rdnam_annotators": {"evaluator_name": "RDNAM", "use_multiple_annotators": True},
    "domain_expert": {"evaluator_name": "domain_expert", "expert_in": "web search"},
    "reasoner_0_3": {"evaluator_name": "reasoner", "relevance_grades": TREC_DL_GRADES},
    "rdnam_0_3": {"evaluator_name": "RDNAM", "relevance_grades": TREC_DL_GRADES},
    "domain_expert_0_3": {
        "evaluator_name": "domain_expert",
        "expert_in": "web search",
        "relevance_grades": TREC_DL_GRADES,
    },
    "reasoner_topical": {"evaluator_name": "reasoner", "relevance_grades": TOPICAL_GRADES},
    "reasoner_strict": {"evaluator_name": "reasoner", "guidelines": LOWER_WHEN_UNSURE},
    "jev_boolean": {"evaluator_name": "jev"},
    "jev_boolean_helps": {
        "evaluator_name": "jev",
        "system_prompt": "Does the document contain information that helps answer the user question?",
    },
    "jev_score": {"evaluator_name": "jev", "boolean_question": False},
    "jev_score_topical": {"evaluator_name": "jev", "boolean_question": False, "relevance_grades": TOPICAL_GRADES},
    "jev_score_0_3": {"evaluator_name": "jev", "boolean_question": False, "relevance_grades": TREC_DL_GRADES},
    "jev_rdnam": {"evaluator_name": "jev_rdnam"},
    "jev_rdnam_aspects": {"evaluator_name": "jev_rdnam", "use_aspects": True},
}
