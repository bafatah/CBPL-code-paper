from __future__ import annotations


PUMP_SCHEDULING_RULES = (
    "Pump-group operating rule: PUMP_11=Pump 11, PUMP_12=Pump 12, PUMP_13=Pump 2C, PUMP_21=AFT-A, PUMP_22=AFT-B, PUMP_23=AFT-C.",
    "There are five reference pump-group grades, and both desulfurization strength and energy use increase with the grade:",
    "Grade 1: if load < 320 and inlet flue-gas concentration < 1000, use Pump 12 + Pump 2C and keep the others off.",
    "Grade 2: if load < 500 and inlet SO2 < 1900, use Pump 11 + Pump 2C + AFT-B and keep the others off.",
    "Grade 3: if load < 550 and inlet SO2 < 2200, use Pump 11 + Pump 2C + AFT-A + AFT-B and keep the others off.",
    "Grade 4: if load < 600 and inlet SO2 < 2400, use Pump 11 + Pump 12 + Pump 2C + AFT-A + AFT-B and keep the others off.",
    "Grade 5: if load > 500 and inlet SO2 > 2400, use all six pumps.",
    "Apply these thresholds flexibly: one variable can be somewhat higher if the other is lower and the overall operating burden still matches the grade.",
    "Three recommendation cases are mutually exclusive. Once one case applies, do not evaluate the other two:",
    "Case 1: if outlet SO2 maximum or predicted value exceeds 35, or measured outlet SO2 mean exceeds 30, strengthen by exactly one grade even if the normal grade conditions are not met.",
    "Case 2: if outlet SO2 is below 10 and the current pump group has surplus desulfurization margin, reduce by exactly one grade to save energy, but only if the lower grade fully satisfies all of its own conditions.",
    "Case 3: if outlet SO2 is greater than 10 and less than 30, keep the current pump group unchanged.",
    "Primary objective: prevent outlet SO2 exceedance first, then save energy where safe.",
    "If the current pump combination does not exactly match a standard grade, infer the current grade from the number of running pumps: 2 pumps=grade 1, 3=grade 2, 4=grade 3, 5=grade 4, 6=grade 5.",
)


def build_system_prompt() -> str:
    rule_lines = "\n".join(f"- {rule}" for rule in PUMP_SCHEDULING_RULES)
    return (
        "You are an expert WFGD pump scheduling analyst.\n"
        "Apply the following pump scheduling rules exactly and produce only a JSON object.\n\n"
        "Rules:\n"
        f"{rule_lines}\n\n"
        "Return JSON only with this schema:\n"
        '{'
        '"reasoning": "short explanation grounded in the rules", '
        '"decision": "increase|keep|decrease", '
        '"target_grade": 1, '
        '"target_pumps": 2, '
        '"one_sentence_reason": "single-sentence justification", '
        '"final_decision_text": "human-readable final decision"'
        '}'
    )
