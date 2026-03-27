from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class Lesson:
    text: str
    weight: int = 2


@dataclass(frozen=True)
class LessonUpdate:
    candidate: str
    op: str


class Guidebook:
    def __init__(self) -> None:
        self._lessons: list[Lesson] = []

    @property
    def lessons(self) -> list[Lesson]:
        return list(self._lessons)

    def apply(self, update: LessonUpdate) -> Lesson:
        op = update.op.upper()
        lesson = self._find_similar(update.candidate)
        if op == "ADD" or lesson is None:
            if lesson is None:
                lesson = Lesson(text=update.candidate, weight=2)
                self._lessons.append(lesson)
            else:
                lesson.weight += 1
            return self._snapshot(lesson)
        if op == "EDIT":
            lesson.text = update.candidate
            lesson.weight += 1
            return self._snapshot(lesson)
        if op == "UPGRADE":
            lesson.weight += 1
            return self._snapshot(lesson)
        if op == "DOWNGRADE":
            lesson.weight -= 1
            if lesson.weight <= 0:
                self._lessons.remove(lesson)
                return Lesson(text=update.candidate, weight=0)
            return self._snapshot(lesson)
        raise ValueError(f"Unsupported lesson update op: {update.op}")

    def render(self) -> str:
        if not self._lessons:
            return "- No lessons consolidated yet."
        ordered = sorted(self._lessons, key=lambda lesson: lesson.weight, reverse=True)
        return "\n".join(f"- ({lesson.weight}) {lesson.text}" for lesson in ordered)

    def _find_similar(self, candidate: str) -> Lesson | None:
        best: Lesson | None = None
        best_score = 0.0
        for lesson in self._lessons:
            score = SequenceMatcher(a=lesson.text, b=candidate).ratio()
            if score > best_score:
                best = lesson
                best_score = score
        return best if best_score >= 0.55 else None

    def _snapshot(self, lesson: Lesson) -> Lesson:
        return Lesson(text=lesson.text, weight=lesson.weight)
