"""The six memory / trust-state representations given to the supervisor (Section 3,
'Intervention conditions'), plus construction of the failure-episode ClaimGraph."""
from __future__ import annotations

import json

from .claimgraph import ClaimGraph, ClaimNode, attribute_cause, label_failure
from .config import AGENTS
from .data.schema import Episode
from .trust import explicit_update


def visible_recommendation(ep: Episode) -> str:
    """Identical across the four causal variants of a base task."""
    t = ep.base_task
    return (f"Recommendation from {{agent}} on task {t.task_id}:\n"
            f"  Question: {t.question}\n"
            f"  [c1] The available information supports the answer below.\n"
            f"  [c2] Final answer: {t.wrong_answer}")


def seen_evidence(ep: Episode) -> list[str]:
    """Evidence ids that actually reached the failing agent."""
    t = ep.base_task
    ids = [e.eid for e in t.evidence if e.eid not in ep.withheld_evidence_ids]
    if ep.cause == "tool_failure" and t.source in ("fever", "hotpotqa"):
        ids = [i for i in ids if i not in t.gold_evidence_ids]   # retrieval returned nothing useful
    return ids


TOOL_NAME = {"dbtool": "sql_query", "gsm8k": "calculator", "fever": "retrieval", "hotpotqa": "retrieval"}


def build_failure_graph(ep: Episode) -> ClaimGraph:
    t = ep.base_task
    g = ClaimGraph()
    agent, skill = ep.failing_agent, t.skill
    if ep.routing_note:
        g.events.append({"type": "routing", "detail": ep.routing_note})
    tool_used = t.source == "dbtool" or ep.tool_state != "ok"
    if tool_used:
        g.events.append({"type": "tool_status", "tool": TOOL_NAME[t.source], "status": ep.tool_state})
    g.add(ClaimNode("c1", None, agent, skill, "The available information supports the answer.",
                    evidence=seen_evidence(ep),
                    tool_state=ep.tool_state if tool_used else "n/a", confidence=0.8))
    g.add(ClaimNode("c2", "c1", agent, skill, f"Final answer: {t.wrong_answer}", evidence=[],
                    tool_state="n/a", confidence=0.8))
    if ep.cause == "missing_context":
        g.events.append({"type": "resolution", "detail":
                         f"Evidence {ep.withheld_evidence_ids} was missing from the agent's context and has been restored."})
    elif ep.cause == "tool_failure":
        g.events.append({"type": "resolution", "detail": "Retrieval tool repaired and verified healthy."})
    return g


def raw_transcript(ep: Episode) -> str:
    """Condition 2: the same information, unstructured, as an interaction log."""
    t = ep.base_task
    seen = set(seen_evidence(ep))
    lines = []
    if ep.routing_note:
        lines.append(f"[orchestrator] {ep.routing_note}")
    else:
        lines.append(f"[orchestrator] Delegated {t.skill} task {t.task_id} to {ep.failing_agent}.")
    ev = "\n".join(f"    ({e.eid}) {e.title}: {e.text}" for e in t.evidence
                   if e.eid in seen and e.eid != "tool_result")
    lines.append(f"[context given to {ep.failing_agent}]\n    Question: {t.question}\n{ev}")
    if t.source == "dbtool":
        if ep.tool_state == "stale":
            lines.append(f"[tool] sql_query -> Result: {t.tool['stale_result']} (served from cache snapshot)")
        elif "tool_result" in seen:
            lines.append(f"[tool] sql_query -> Result: {t.tool['fresh_result']}")
        else:
            lines.append("[orchestrator] sql_query result was not forwarded to the agent.")
    elif ep.tool_state == "error_empty":
        lines.append("[tool] retrieval -> HTTP 503, returned 0 documents")
    elif ep.tool_state == "stale":
        lines.append("[tool] calculator -> returned value from a cached expression (cache key collision)")
    lines.append("[" + ep.failing_agent + "] " + visible_recommendation(ep).format(agent=ep.failing_agent))
    lines.append("[supervisor] Recommendation REJECTED: answer is incorrect.")
    if ep.cause == "missing_context":
        lines.append("[system] context sync completed; previously unavailable documents now indexed.")
    if ep.cause == "tool_failure":
        lines.append("[system] tool service restored; health check passed.")
    return "\n".join(lines)


def render_memory(condition: str, ep: Episode, graph: ClaimGraph, attribution: dict,
                  prior: dict, ablation: str | None = None, observed_rel: float | None = None) -> dict:
    """Returns {'text': memory shown to supervisor, 'recommended_trust': dict|None}."""
    agent, skill = ep.failing_agent, ep.affected_skill
    if condition == "no_memory":
        return {"text": "(No record of past incidents is retained.)", "recommended_trust": None}
    if condition == "raw_history":
        return {"text": raw_transcript(ep), "recommended_trust": None}
    if condition == "outcome_only":
        return {"text": f"Reputation ledger: {agent} -> 1 failure (recommendation rejected).",
                "recommended_trust": None}
    if condition == "skill_conditioned":
        return {"text": f"Reputation ledger: {agent} -> 1 failure on skill '{skill}'.",
                "recommended_trust": None}
    drop = {ablation} if ablation else set()
    text = "Causal ClaimGraph for the rejected recommendation:\n" + graph.render(drop)
    rec = None
    if condition == "claimgraph_update" and ablation != "no_update_rule":
        rec = explicit_update(prior, attribution, observed_rel)
        text += ("\n\nScoped trust-update rule output (recommended trust for " + agent + "):\n"
                 + json.dumps(rec))
    return {"text": text, "recommended_trust": rec}


def run_attribution(ep: Episode, graph: ClaimGraph) -> dict:
    att = attribute_cause(graph, "c2", ep.failing_agent, ep.affected_skill,
                          ep.base_task.gold_evidence_ids, seen_evidence(ep))
    label_failure(graph, "c2", att)
    return att
