"""Reciprocal Rank Fusion (Cormack et al. 2009). A document appearing in
multiple retrieval paths outranks one appearing in only one."""


def rrf(result_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for lst in result_lists:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
