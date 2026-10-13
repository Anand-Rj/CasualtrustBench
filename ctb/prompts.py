"""Prompt templates. All supervisor prompts are identical across conditions except for the
memory block, so the memory representation is the only manipulated input."""
from __future__ import annotations

import json

from .config import AGENTS, SKILLS
from .data.schema import Task

SUBAGENT_SYSTEM = ("You are {agent}, a persistent specialist subagent in an LLM agent team. "
                   "Your declared specialty is {specialty}. Answer only from the information "
                   "provided. Reply with JSON only.")

ANSWER_SPEC = {
    "label": 'the answer must be exactly "SUPPORTS" or "REFUTES"',
    "short": "the answer must be a short span (an entity, name, or yes/no)",
    "number": "the answer must be a single number with no units",
}


def subagent_prompt(task: Task, evidence_ids: list[str], tool_line: str | None) -> str:
    ev = "\n".join(f"({e.eid}) {e.title}: {e.text}" for e in task.evidence
                   if e.eid in evidence_ids and e.eid != "tool_result")
    tool = f"\nTool output:\n{tool_line}\n" if tool_line else ""
    return (f"Task ({task.skill}): {task.question}\n\nEvidence:\n{ev or '(none)'}\n{tool}\n"
            f"Return JSON: {{\"reasoning\": str, \"claims\": [{{\"text\": str, \"evidence\": [ids]}}], "
            f"\"answer\": str, \"confidence\": float}}; {ANSWER_SPEC[task.answer_format]}.")


SUPERVISOR_SYSTEM = ("You are the Supervisor/Orchestrator of a persistent LLM agent team with "
                     "four specialist subagents. You decide how much to trust each subagent on "
                     "each skill and whom to delegate future tasks to. Reply with JSON only.")


def team_block(public: dict) -> str:
    rows = [f"- {a} (declared specialty: {AGENTS[a]}): " +
            ", ".join(f"{k}={public[a][k]:.2f}" for k in SKILLS) for a in public]
    return ("Team track record (estimated probability of a correct answer, per skill; "
            "values off an agent's specialty are team averages, i.e. not yet observed):\n"
            + "\n".join(rows))


def prior_prompt(public: dict, agent: str) -> str:
    return (team_block(public) + f"\n\nBefore any new evidence, state your trust in {agent} for "
            f"each skill as the probability it answers a task of that skill correctly.\n"
            f"Return JSON: {{\"trust\": {{{', '.join(f'\"{k}\": float' for k in SKILLS)}}}}}")


def attribute_prompt(public: dict, agent: str, prior: dict, memory_text: str) -> str:
    return (team_block(public) + f"\n\nYour current trust in {agent}: {json.dumps(prior)}\n\n"
            f"A recommendation from {agent} was just rejected as incorrect. "
            f"Here is what your memory system retained about it:\n{memory_text}\n\n"
            "1) Diagnose the failure: cause in {missing_context, tool_failure, out_of_specialty, "
            "persistent_reasoning, unknown}; the originating claim id, agent and source; whether "
            "the cause is transient or persistent; which skills it bears on.\n"
            f"2) Give your UPDATED trust in {agent} for every skill, and a trust score for each claim.\n"
            "Return JSON: {\"cause\": str, \"originating_claim\": str, \"originating_agent\": str, "
            "\"originating_source\": str, \"persistence\": str, \"affected_skills\": [str], "
            f"\"trust\": {{{', '.join(f'\"{k}\": float' for k in SKILLS)}}}, "
            "\"claim_trust\": {\"c1\": float, \"c2\": float}, \"rationale\": str}")


def delegate_prompt(public: dict, agent: str, trust: dict, memory_text: str,
                    probe: Task) -> str:
    return (team_block(public) + f"\n\nYour current trust in {agent} (after a recent incident): "
            f"{json.dumps(trust)}\nMemory of the incident:\n{memory_text}\n\n"
            f"New task (skill: {probe.skill}): {probe.question}\n"
            "Choose which subagent to delegate to and whether to spend extra effort verifying "
            "its answer (verification costs time and tokens; only verify when it is worth it).\n"
            f"Return JSON: {{\"agent\": one of {list(AGENTS)}, \"verify\": bool, \"reason\": str}}")


def verify_prompt(probe: Task, answer: str) -> str:
    ev = "\n".join(f"({e.eid}) {e.title}: {e.text}" for e in probe.evidence)
    return (f"Verify a subagent's answer.\nTask: {probe.question}\nEvidence:\n{ev}\n"
            f"Proposed answer: {answer}\n"
            "Return JSON: {\"verdict\": \"accept\" or \"reject\", \"corrected_answer\": str or null, "
            "\"reason\": str}")
