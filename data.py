from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


FLOAT_RE = r"([-+]?\d+(?:\.\d+)?)"


@dataclass(frozen=True)
class DecisionEpisode:
    episode_id: str
    source_input: str
    source_output: str
    state_summary: str
    current_pumps: int
    total_power_kw: float | None
    load_mw: float | None
    inlet_so2_mg_m3: float | None
    outlet_so2_mg_m3: float | None
    outlet_trend: float | None
    slurry_flow_m3_h: float | None
    ph: float | None
    expert_action: int
    target_pumps: int
    rationale: str


class DatasetEpisodeParser:
    """Parse the JSON supervision dataset into structured decision episodes."""

    def parse_record(self, record: dict[str, Any], *, index: int) -> DecisionEpisode:
        source_input = str(record.get("input", "")).strip()
        source_output = str(record.get("output", "")).strip()
        current_pumps = int(self._extract_required(source_input, r"当前配置[^\n:]*:\s*(?:自定义\()?(\d+)\s*(?:台)?泵\)?"))
        total_power = self._extract_optional_float(source_input, rf"总功率[^\n:]*:\s*{FLOAT_RE}\s*kW")
        load_mw = self._extract_optional_float(source_input, rf"负荷[^\n:]*:\s*{FLOAT_RE}\s*MW")
        inlet_so2 = self._extract_optional_float(source_input, rf"入口SO[₂2][^\n:]*:\s*{FLOAT_RE}\s*mg/m")
        outlet_so2 = self._extract_optional_float(source_input, rf"出口SO[₂2][^\n:]*:\s*{FLOAT_RE}\s*mg/m")
        outlet_trend = self._extract_optional_float(source_input, rf"出口SO[₂2][^\n:]*:\s*{FLOAT_RE}\s*mg/m[^\n]*趋势\s*{FLOAT_RE}", group=2)
        slurry_flow = self._extract_optional_float(source_input, rf"浆液流量[^\n:]*:\s*{FLOAT_RE}\s*m")
        ph = self._extract_optional_float(source_input, rf"石膏PH[^\n:]*:\s*{FLOAT_RE}")
        action, target_pumps = self._parse_action(source_output, current_pumps)
        rationale = self._parse_rationale(source_output)
        episode_id = f"episode-{index:05d}"
        return DecisionEpisode(
            episode_id=episode_id,
            source_input=source_input,
            source_output=source_output,
            state_summary=source_input,
            current_pumps=current_pumps,
            total_power_kw=total_power,
            load_mw=load_mw,
            inlet_so2_mg_m3=inlet_so2,
            outlet_so2_mg_m3=outlet_so2,
            outlet_trend=outlet_trend,
            slurry_flow_m3_h=slurry_flow,
            ph=ph,
            expert_action=action,
            target_pumps=target_pumps,
            rationale=rationale,
        )

    def _extract_required(self, text: str, pattern: str) -> str:
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Could not find required pattern: {pattern}")
        return match.group(1)

    def _extract_optional_float(self, text: str, pattern: str, *, group: int = 1) -> float | None:
        match = re.search(pattern, text)
        if not match:
            return None
        return float(match.group(group))

    def _parse_action(self, output_text: str, current_pumps: int) -> tuple[int, int]:
        final_decision = self._extract_final_decision_text(output_text)
        transition = re.search(r"当前\s*(\d+)\s*台\s*→\s*(\d+)\s*台", final_decision)
        if transition:
            source = int(transition.group(1))
            target = int(transition.group(2))
            if source != current_pumps:
                current_pumps = source
            action = 1 if target > current_pumps else -1 if target < current_pumps else 0
            return action, target
        if "减少" in final_decision or "减泵" in final_decision:
            return -1, current_pumps - 1
        if "增加" in final_decision or "增泵" in final_decision:
            return 1, current_pumps + 1
        keep_match = re.search(r"维持当前泵数（(\d+)\s*台）", final_decision)
        if keep_match:
            return 0, int(keep_match.group(1))
        return 0, current_pumps

    def _parse_rationale(self, output_text: str) -> str:
        match = re.search(r"一句话理由\s*\n(.+?)(?:\n\s*\n|\n##|\Z)", output_text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in output_text.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def _extract_final_decision_text(self, output_text: str) -> str:
        match = re.search(r"最终决策[^\n]*\n(.+?)(?:\n\s*\n|\Z)", output_text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return output_text


def load_dataset(path: str | Path, *, limit: int | None = None) -> list[DecisionEpisode]:
    parser = DatasetEpisodeParser()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items: Iterable[dict[str, Any]]
    if isinstance(data, list):
        items = data[:limit] if limit is not None else data
    else:
        raise ValueError("Expected a list of dataset records.")
    return [parser.parse_record(record, index=index) for index, record in enumerate(items)]
