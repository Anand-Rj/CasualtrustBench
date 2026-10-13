"""Causal ClaimGraph: claim-level provenance extended with failure cause, persistence,
affected scope and resolution state (Section 3 of the proposal).

Agent trust and claim trust are kept separate: a capable agent can emit a low-trust
claim (e.g. after a retrieval failure) and vice versa.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from .config import AGENTS, NEARBY, TRANSIENT_CAUSES


@dataclass
class ClaimNode:
    claim_id: str
    parent_claim_id: str | None
    agent: str
    skill: str
    claim_text: str
    evidence: list[str] = field(default_factory=list)
    tool_state: str = "n/a"           # ok | error_empty | stale | n/a
    confidence: float | None = None
    verification: str = "unverified"  # unverified | accepted | rejected
    failure_cause: str | None = None  # missing_context | tool_failure | out_of_specialty | persistent_reasoning
    persistence: str | None = None    # transient | persistent
    scope: list[str] = field(default_factory=list)
    resolved: bool | None = None


@dataclass
class ClaimGraph:
    nodes: dict[str, ClaimNode] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)   # resolution / routing events

    def add(self, node: ClaimNode) -> None:
        self.nodes[node.claim_id] = node

    def lineage(self, claim_id: str) -> list[str]:
        out, cur = [], claim_id
        while cur:
            out.append(cur)
            cur = self.nodes[cur].parent_claim_id if cur in self.nodes else None
        return out

    def root_cause_node(self, claim_id: str) -> ClaimNode:
        """Earliest ancestor that carries a failure label (or the claim itself)."""
        nodes = [self.nodes[c] for c in self.lineage(claim_id)]
        labelled = [n for n in nodes if n.failure_cause]
        return labelled[-1] if labelled else nodes[-1]

    def render(self, drop: set[str] | None = None) -> str:
        """JSON view shown to the supervisor. `drop` removes fields (ablations)."""
        drop = drop or set()
        rows = []
        for n in self.nodes.values():
            d = asdict(n)
            if "no_parent_links" in drop:
                d.pop("parent_claim_id")
            if "no_tool_state" in drop:
                d.pop("tool_state")
            if "no_cause_label" in drop:
                d.pop("failure_cause")
            if "no_persistence_resolution" in drop:
                d.pop("persistence"); d.pop("resolved")
            rows.append(d)
        ev = self.events
        if "no_persistence_resolution" in drop:
            ev = [e for e in ev if e.get("type") != "resolution"]
        if "no_tool_state" in drop:
            ev = [e for e in ev if e.get("type") != "tool_status"]
        return json.dumps({"claims": rows, "events": ev}, indent=1)


# --------------------------------------------------------------------------- attribution
def attribute_cause(graph: ClaimGraph, final_claim_id: str, agent: str, skill: str,
                    required_evidence: list[str], seen_evidence: list[str]) -> dict:
    """Rule-based Failure-Cause Attributor working only from recorded provenance.

    It does not read the benchmark's ground-truth label; it infers the cause from the
    tool state, evidence coverage and routing recorded in the graph.
    """
    nodes = [graph.nodes[c] for c in graph.lineage(final_claim_id)]
    if any(n.tool_state in ("error_empty", "stale") for n in nodes):
        cause = "tool_failure"
    elif set(required_evidence) - set(seen_evidence):
        cause = "missing_context"
    elif AGENTS.get(agent) != skill:
        cause = "out_of_specialty"
    else:
        cause = "persistent_reasoning"
    persistent = cause not in TRANSIENT_CAUSES
    resolved = any(e.get("type") == "resolution" for e in graph.events) if not persistent else False
    if cause == "persistent_reasoning":
        scope = [skill, NEARBY[skill]]
    else:
        scope = [skill]
    return {"cause": cause, "persistence": "persistent" if persistent else "transient",
            "scope": scope, "resolved": resolved}


def label_failure(graph: ClaimGraph, final_claim_id: str, attribution: dict) -> None:
    root = graph.nodes[final_claim_id]
    root.failure_cause = attribution["cause"]
    root.persistence = attribution["persistence"]
    root.scope = attribution["scope"]
    root.resolved = attribution["resolved"]
    root.verification = "rejected"
