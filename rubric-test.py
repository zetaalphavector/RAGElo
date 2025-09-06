import hashlib
import json
import os

from ragelo import Experiment, get_answer_evaluator, get_llm_provider, get_retrieval_evaluator
from ragelo.types.evaluables import PairwiseGame

data_home = "/Users/acamara/zav/research/agents"

agents = ["HRA_4o", "HRA_4o_mini"]
experiment = Experiment("4o-vs-4o-mini-HRA", verbose=True)
# experiment = Experiment("4o-vs-4o-mini-HRA", verbose=True)
# experiment = Experiment("4o-vs-4o-mini-HRA_4.1-mini", verbose=True)
# experiment = Experiment("4o-vs-4o-mini-HRA_no_docs_4.1-mini", verbose=True)
# experiment = Experiment("4o-vs-4o-mini-HRA_no_docs_4o", verbose=True)
# experiment = Experiment("4o-vs-4o-mini-HRA_no_docs_o4-mini", verbose=True)
# experiment = Experiment("4o-vs-4o-mini-HRA_no_docs_o3", verbose=True)


def hash_document(doc_url: str) -> str:
    return hashlib.sha256(doc_url.encode()).hexdigest()[:8]


def find_overlap(text1: str, text2: str, min_overlap: int = 10) -> tuple[int, int, bool]:
    """
    Find overlap between two texts.
    Returns (overlap_length, position, is_text1_first) where:
    - overlap_length: length of overlap in characters
    - position: where the overlap starts in the first text
    - is_text1_first: True if text1 comes before text2, False otherwise
    """
    max_overlap = 0
    best_pos = 0
    text1_first = True

    # Check if text1 suffix overlaps with text2 prefix
    for i in range(min_overlap, min(len(text1), len(text2)) + 1):
        if text1[-i:].lower() == text2[:i].lower():
            if i > max_overlap:
                max_overlap = i
                best_pos = len(text1) - i
                text1_first = True

    # Check if text2 suffix overlaps with text1 prefix
    for i in range(min_overlap, min(len(text1), len(text2)) + 1):
        if text2[-i:].lower() == text1[:i].lower():
            if i > max_overlap:
                max_overlap = i
                best_pos = len(text2) - i
                text1_first = False

    return max_overlap, best_pos, text1_first


def combine_texts(text1: str, text2: str) -> str:
    """Combine two texts if they overlap, otherwise return None."""
    overlap_length, overlap_pos, text1_first = find_overlap(text1, text2)

    if overlap_length > 0:
        if text1_first:
            # text1 comes first, text2 continues it
            return text1 + text2[overlap_length:]
        else:
            # text2 comes first, text1 continues it
            return text2 + text1[overlap_length:]

    return None


def try_combine_with_existing(new_text: str, existing_texts: list[str]) -> list[str]:
    """
    Try to combine new_text with existing texts.
    Returns updated list of texts.
    """
    for i, existing_text in enumerate(existing_texts):
        combined = combine_texts(new_text, existing_text)
        if combined:
            # Replace the existing text with the combined version
            updated_texts = existing_texts.copy()
            updated_texts[i] = combined
            return updated_texts

    # No combination possible, add as new text
    return existing_texts + [new_text]


# Change doc_contents to store lists of text snippets per hashed URL
doc_contents = {}
doc_authors: dict[str, str] = {}
doc_to_query_mapping: dict[str, set[str]] = {}
doc_titles: dict[str, str] = {}

with open(os.path.join(data_home, "queries.json"), "r") as f:
    data = json.load(f)
    for query_id, query in data.items():
        experiment.add_query(query, query_id=query_id, exists_ok=True)
        for agent in agents:
            with open(os.path.join(data_home, "answers", agent, f"{query_id}_citations.json"), "r") as f:
                data = json.load(f)
                answer = data["cleaned_answer"]
                for doc in data["citations"]:
                    for instance in doc["instances"]:
                        hashed_url = hash_document(instance["url"])
                        quote = instance["quote"]
                        if hashed_url not in doc_to_query_mapping:
                            doc_to_query_mapping[hashed_url] = set()
                            doc_titles[hashed_url] = doc["title"]
                            doc_authors[hashed_url] = doc["authors"]
                        doc_to_query_mapping[hashed_url].add(query_id)

                        if hashed_url not in doc_contents:
                            doc_contents[hashed_url] = [quote]
                        else:
                            # Try to combine with existing texts
                            doc_contents[hashed_url] = try_combine_with_existing(quote, doc_contents[hashed_url])
                    answer = answer.replace(f"[[{doc['number']}]]", f"[[{hashed_url}]]")
            experiment.add_agent_answer(answer, query_id=query_id, agent=agent)

for hashed_url, texts in doc_contents.items():
    if len(texts) > 1:
        doc_contents[hashed_url] = "(...)".join(texts)
    else:
        doc_contents[hashed_url] = texts[0]
    doc_contents[hashed_url] = (
        f"Title: {doc_titles[hashed_url]}\nAuthors: {' and '.join(doc_authors[hashed_url])}\n{doc_contents[hashed_url]}"
    )
    for query_id in doc_to_query_mapping[hashed_url]:
        experiment.add_retrieved_doc(doc_contents[hashed_url], doc_id=hashed_url, query_id=query_id)
experiment.save()

llm_provider_41 = get_llm_provider("openai", model="gpt-4.1-mini")
llm_provider = get_llm_provider("openai", model="gpt-5-mini")

use_retrieved_docs = False
retrieval_evaluator = get_retrieval_evaluator(
    "domain_expert",
    expert_in="AI and Computer Science",
    n_processes=20,
    rich_print=True,
    llm_provider=llm_provider_41,
)
answer_evaluator = get_answer_evaluator(
    "rubric_pairwise",
    llm_provider=llm_provider,
    expert_in="AI and Computer Science",
    rich_print=True,
    bidirectional=True,
    n_processes=1,
    document_relevance_threshold=1,
    force=True,
)

retrieval_evaluator.evaluate_experiment(experiment)
answer_evaluator.evaluate_experiment(experiment)


def get_winner_name(game: PairwiseGame) -> str:
    if game.evaluation.answer["winner"] == "A":
        return game.agent_a_answer.agent
    elif game.evaluation.answer["winner"] == "B":
        return game.agent_b_answer.agent
    else:
        return "TIE"


combined_answers = {}
missmatches = 0
reversals = 0
comparisons = 0

for query in experiment:
    for game in query.pairwise_games:
        if game.evaluation is None:
            continue
        if query.qid not in combined_answers:
            combined_answers[query.qid] = get_winner_name(game)
            continue
        comparisons += 1
        new_winner = get_winner_name(game)
        current_winner = combined_answers[query.qid]
        if current_winner != new_winner:
            missmatches += 1
            if current_winner == "TIE":
                combined_answers[query.qid] = new_winner
            elif new_winner == "TIE":
                combined_answers[query.qid] = current_winner
            else:
                combined_answers[query.qid] = "TIE"
                reversals += 1


print("Rubric evaluator results:")
print("LLM model:", llm_provider.config.model)
print("Use retrieved docs:", use_retrieved_docs)
per_agent_wins_total = {"HRA_4o": 0, "HRA_4o_mini": 0, "TIE": 0}
for qid in combined_answers:
    per_agent_wins_total[combined_answers[qid]] += 1

print(f"HRA_4o wins: {per_agent_wins_total['HRA_4o']}")
print(f"HRA_4o_mini wins: {per_agent_wins_total['HRA_4o_mini']}")
print(f"Ties: {per_agent_wins_total['TIE']}")
print(f"Missmatches: {missmatches}")
print(f"Reversals: {reversals}")

# def get_winner_name(game: PairwiseGame, criteria: str) -> str:
#     if game.evaluation.answer[criteria]["judgement"] == "Answer A is superior to answer B":
#         return game.agent_a_answer.agent
#     elif game.evaluation.answer[criteria]["judgement"] == "Answer B is superior to answer A":
#         return game.agent_b_answer.agent
#     else:
#         return "TIE"


# combined_answers = {}
# new_ties = 0
# comparisons = 0
# for query in experiment:
#     combined_answers[query.qid] = {}
#     for game in query.pairwise_games:
#         if game.evaluation is None:
#             continue
#         for criteria in game.evaluation.answer:
#             if criteria == "winner":
#                 continue
#             if criteria not in combined_answers[query.qid]:
#                 combined_answers[query.qid][criteria] = get_winner_name(game, criteria)
#                 continue
#             comparisons += 1
#             new_winner = get_winner_name(game, criteria)
#             current_winner = combined_answers[query.qid][criteria]
#             if current_winner != new_winner:
#                 if current_winner == "TIE":
#                     combined_answers[query.qid][criteria] = new_winner
#                 elif new_winner == "TIE":
#                     combined_answers[query.qid][criteria] = current_winner
#                 else:
#                     combined_answers[query.qid][criteria] = "TIE"
#                     new_ties += 1

# print(f"New ties: {new_ties}")
# print(f"Comparisons: {comparisons}")


# agents += ["TIE"]
# per_agent_wins_criteria = {agent: 0 for agent in agents}
# per_agent_wins_total = {agent: 0 for agent in agents}
# for query in combined_answers:
#     per_agent_wins_query = {agent: 0 for agent in agents}
#     for criteria in combined_answers[query]:
#         winner = combined_answers[query][criteria]
#         per_agent_wins_query[winner] += 1
#         per_agent_wins_criteria[winner] += 1
#     most_winner = sorted(per_agent_wins_query.items(), key=lambda x: x[1], reverse=True)[0][0]
#     per_agent_wins_total[most_winner] += 1

# print(f"HRA_4o wins: {per_agent_wins_total['HRA_4o']}")
# print(f"HRA_4o_mini wins: {per_agent_wins_total['HRA_4o_mini']}")
# print(f"Ties: {per_agent_wins_total['TIE']}")

# TODO: Get the best agent and optmize it using TexGrad/DsPy.
