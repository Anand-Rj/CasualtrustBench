from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Evidence:
    eid: str
    title: str
    text: str


@dataclass
class Task:
    """One base task drawn from a source dataset."""
    task_id: str
    source: str              # fever | hotpotqa | gsm8k | dbtool
    skill: str
    question: str
    evidence: list[Evidence]
    gold_answer: str
    gold_evidence_ids: list[str]          # evidence the correct answer needs
    wrong_answer: str                     # the fixed target incorrect recommendation r-
    answer_format: str = "short"          # label | short | number
    tool: dict[str, Any] | None = None    # for tool tasks: {"db": <sql setup>, "query_hint": ...}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        d = dict(d)
        d["evidence"] = [Evidence(**e) for e in d["evidence"]]
        return cls(**d)


@dataclass
class Episode:
    """A matched failure episode: base task x causal condition."""
    episode_id: str
    base_task: Task
    cause: str
    failing_agent: str
    affected_skill: str
    persistent: bool
    split: str
    withheld_evidence_ids: list[str] = field(default_factory=list)
    tool_state: str = "ok"                # ok | error_empty | stale
    routing_note: str = ""
    probes: list[dict] = field(default_factory=list)   # [{"task": Task, "slot": affected|nearby|unrelated}]
