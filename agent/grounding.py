"""Numeric grounding: every number in the final answer must come from an executed query result.

The LLM may round (4.12 % for 0.041203) or convert a fraction to a percentage, but it may not invent
or mentally compute a number. Numbers that also appear in the question, years, and small counting
words (0 to 10) are always allowed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# 5,575 | 5 575 | 4.12 | -0.5 | 53% (the % is captured separately)
_NUM = re.compile(r"(?<![\w.])([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(\s?%)?")


@dataclass
class NumberMention:
    text: str
    value: float
    decimals: int
    is_percent: bool


@dataclass
class GroundingReport:
    mentions: list[NumberMention] = field(default_factory=list)
    ungrounded: list[NumberMention] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.ungrounded

    def to_dict(self) -> dict:
        return {"n_numbers": len(self.mentions), "ungrounded": [m.text for m in self.ungrounded]}


def extract_numbers(text: str) -> list[NumberMention]:
    out = []
    # Ignore identifiers such as q3, EU27_2020, DIE_X_HYB and ISO dates' separators.
    cleaned = re.sub(r"\b[A-Za-z_]+\d[\w]*\b", " ", text)
    for m in _NUM.finditer(cleaned):
        raw = m.group(1)
        num = raw.replace(",", "")
        decimals = len(num.split(".")[1]) if "." in num else 0
        out.append(NumberMention(text=m.group(0).strip(), value=float(num), decimals=decimals, is_percent=bool(m.group(2))))
    return out


def _candidates(values: Iterable[Any]) -> list[float]:
    out = []
    for v in values:
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
            out.append(float(v) * 100.0)  # fraction shown as a percentage
    return out


def _matches(m: NumberMention, cands: list[float]) -> bool:
    tol = 0.5 * 10 ** (-m.decimals) + 1e-9
    target = abs(m.value)
    return any(abs(abs(c) - target) <= tol for c in cands)


def check_grounding(answer: str, result_values: Iterable[Any], question: str = "") -> GroundingReport:
    cands = _candidates(result_values)
    question_numbers = {n.value for n in extract_numbers(question)}
    report = GroundingReport(mentions=extract_numbers(answer))
    for m in report.mentions:
        if m.value in question_numbers:
            continue
        if m.decimals == 0 and not m.is_percent and (0 <= m.value <= 10 or 1990 <= m.value <= 2100):
            continue
        if not _matches(m, cands):
            report.ungrounded.append(m)
    return report
