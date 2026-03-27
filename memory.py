from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from cbpl_paper.data import DecisionEpisode


@dataclass(frozen=True)
class CaseRecord:
    episode_id: str
    state_summary: str
    action: int
    target_pumps: int
    reasoning: str
    outcome_success: bool | None = None
    expert_correction: int | None = None

    @classmethod
    def from_episode(cls, episode: DecisionEpisode, *, reasoning: str | None = None) -> "CaseRecord":
        return cls(
            episode_id=episode.episode_id,
            state_summary=episode.state_summary,
            action=episode.expert_action,
            target_pumps=episode.target_pumps,
            reasoning=reasoning or episode.rationale,
            outcome_success=True,
            expert_correction=episode.expert_action,
        )


@dataclass(frozen=True)
class RetrievalHit:
    case: CaseRecord
    score: float


class CaseRetriever:
    def __init__(self) -> None:
        self._cases: list[CaseRecord] = []
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] = []

    def fit(self, cases: list[CaseRecord]) -> None:
        self._cases = list(cases)
        tokenized = [self._tokenize(case.state_summary) for case in self._cases]
        df = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        total_docs = max(len(tokenized), 1)
        self._idf = {token: math.log((1 + total_docs) / (1 + freq)) + 1.0 for token, freq in df.items()}
        self._vectors = [self._vectorize(tokens) for tokens in tokenized]

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        query_vector = self._vectorize(self._tokenize(query))
        query_text = self._normalize(query)
        hits = [
            RetrievalHit(case=case, score=self._adjusted_score(query_text, case.state_summary, query_vector, vector))
            for case, vector in zip(self._cases, self._vectors)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        normalized = self._normalize(text)
        compact = re.sub(r"\s+", "", normalized)
        ngrams: list[str] = []
        for size in range(2, 5):
            ngrams.extend(compact[index : index + size] for index in range(max(len(compact) - size + 1, 0)))
        return ngrams or [compact] if compact else []

    def _normalize(self, text: str) -> str:
        normalized = text.lower()
        normalized = normalized.replace("so₂", "so2")
        normalized = normalized.replace("m³", "m3")
        normalized = normalized.replace("→", "->")
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", normalized)

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        return {token: (count / total) * self._idf.get(token, 1.0) for token, count in counts.items()}

    def _cosine(self, left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        numerator = sum(value * right.get(token, 0.0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _adjusted_score(
        self,
        query_text: str,
        case_text: str,
        query_vector: dict[str, float],
        case_vector: dict[str, float],
    ) -> float:
        score = self._cosine(query_vector, case_vector)
        normalized_case = self._normalize(case_text)
        if "上升" in query_text and "下降" in normalized_case:
            score -= 0.08
        if "下降" in query_text and "上升" in normalized_case:
            score -= 0.08
        return score
