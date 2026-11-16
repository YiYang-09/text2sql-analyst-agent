"""Load the benchmark questions and compute gold answers."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import yaml

from agent.db import Database, QueryResult

ROOT = pathlib.Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "bench" / "questions.yaml"


@dataclass
class Question:
    id: str
    split: str
    difficulty: str
    skills: list[str]
    question: str
    sql: str
    order_matters: bool = False
    notes: str = ""
    gold: QueryResult | None = field(default=None, repr=False)


def load_questions(path: pathlib.Path = QUESTIONS, split: str | None = None) -> list[Question]:
    items = yaml.safe_load(path.read_text(encoding="utf-8"))
    qs = [Question(**{k: v for k, v in it.items()}) for it in items]
    ids = [q.id for q in qs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids")
    return [q for q in qs if split is None or q.split == split]


def attach_gold(questions: list[Question], db: Database) -> list[Question]:
    for q in questions:
        q.gold = db.run(q.sql)
        if not q.gold.ok:
            raise ValueError(f"Gold SQL for {q.id} failed: {q.gold.error}")
    return questions
