from __future__ import annotations

from cbpl_paper.guidebook import Guidebook
from cbpl_paper.memory import CaseRecord
from cbpl_paper.rules import GradeRuleBook


class PromptComposer:
    def __init__(self, *, grade_book: GradeRuleBook | None = None) -> None:
        self.grade_book = grade_book or GradeRuleBook.default()

    def compose(self, *, current_state: str, retrieved_cases: list[CaseRecord], guidebook: Guidebook) -> str:
        seed_rules = "\n".join(f"- {rule}" for rule in self.grade_book.seed_rules())
        cases = self._format_cases(retrieved_cases)
        return (
            "You are the CBPL decision module for WFGD pump scheduling.\n\n"
            "Seed Safety Rules\n"
            f"{seed_rules}\n\n"
            "Lesson Guidebook\n"
            f"{guidebook.render()}\n\n"
            "Retrieved Cases\n"
            f"{cases}\n\n"
            "Current State\n"
            f"{current_state}\n\n"
            "Response Format\n"
            "- reasoning: short explanation grounded in rules and retrieved cases\n"
            "- proposed_action: one of -1, 0, +1\n"
            "- expected_observation: concise post-action expectation"
        )

    def _format_cases(self, cases: list[CaseRecord]) -> str:
        if not cases:
            return "- No retrieved cases."
        return "\n".join(
            f"- {case.episode_id}: action {case.action:+d}, target {case.target_pumps} pumps, reasoning: {case.reasoning}"
            for case in cases
        )
