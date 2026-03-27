from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Protocol

from cbpl_paper.data import DecisionEpisode
from cbpl_paper.guidebook import Guidebook
from cbpl_paper.memory import CaseRecord, CaseRetriever
from cbpl_paper.prompting import PromptComposer
from cbpl_paper.qwen import QwenCBPLClient, QwenCBPLConfig, QwenCBPLDecider
from cbpl_paper.rules import RuleGate, SeedRulePolicy


@dataclass(frozen=True)
class CBPLDecision:
    prompt: str
    proposed_action: int
    action: int
    explanation: str
    retrieved_cases: list[CaseRecord]


class Decider(Protocol):
    def decide(
        self,
        *,
        episode: DecisionEpisode,
        prompt: str,
        retrieved_cases: list[CaseRecord],
        seed_action: int,
    ) -> tuple[int, str]:
        ...


class HeuristicCBPLDecider:
    """A local stand-in that uses retrieved cases plus seed rules without calling an external LLM."""

    def decide(
        self,
        *,
        episode: DecisionEpisode,
        prompt: str,
        retrieved_cases: list[CaseRecord],
        seed_action: int,
    ) -> tuple[int, str]:
        if retrieved_cases:
            avg = mean(case.action for case in retrieved_cases)
            if avg > 0.25:
                return 1, "Retrieved cases favor strengthening the pump level."
            if avg < -0.25:
                return -1, "Retrieved cases favor reducing the pump level."
        return seed_action, "Falling back to the seed safety rules."


class CBPLSystem:
    def __init__(
        self,
        *,
        retriever: CaseRetriever | None = None,
        guidebook: Guidebook | None = None,
        prompt_composer: PromptComposer | None = None,
        rule_gate: RuleGate | None = None,
        seed_policy: SeedRulePolicy | None = None,
        decider: Decider | None = None,
    ) -> None:
        self.retriever = retriever or CaseRetriever()
        self.guidebook = guidebook or Guidebook()
        self.prompt_composer = prompt_composer or PromptComposer()
        self.rule_gate = rule_gate or RuleGate()
        self.seed_policy = seed_policy or SeedRulePolicy()
        self.decider = decider or HeuristicCBPLDecider()

    @classmethod
    def from_env(cls) -> "CBPLSystem":
        config = QwenCBPLConfig.from_env()
        if config is None:
            return cls()
        decider = QwenCBPLDecider(client=QwenCBPLClient(config=config))
        return cls(decider=decider)

    def bootstrap(self, episodes: list[DecisionEpisode]) -> None:
        cases = [CaseRecord.from_episode(episode) for episode in episodes]
        self.retriever.fit(cases)

    def decide(self, episode: DecisionEpisode, *, top_k: int = 5) -> CBPLDecision:
        hits = self.retriever.retrieve(episode.state_summary, top_k=top_k)
        retrieved_cases = [hit.case for hit in hits]
        seed_recommendation = self.seed_policy.recommend(episode)
        prompt = self.prompt_composer.compose(
            current_state=episode.state_summary,
            retrieved_cases=retrieved_cases,
            guidebook=self.guidebook,
        )
        proposed_action, explanation = self.decider.decide(
            episode=episode,
            prompt=prompt,
            retrieved_cases=retrieved_cases,
            seed_action=seed_recommendation.action,
        )
        action = self.rule_gate.project(proposed_action, episode)
        return CBPLDecision(
            prompt=prompt,
            proposed_action=proposed_action,
            action=action,
            explanation=explanation,
            retrieved_cases=retrieved_cases,
        )
