"""Experiment configuration for CausalTrustBench.

Every number that drives compute (tasks, seeds, probes, conditions) lives here so that
`scripts/estimate_compute.py` and the real runner read the same values.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json

SKILLS = ["fact_verification", "multihop", "quantitative", "tool_use"]

# Which skill each source dataset exercises.
SOURCE_SKILL = {
    "fever": "fact_verification",
    "hotpotqa": "multihop",
    "gsm8k": "quantitative",
    "dbtool": "tool_use",
}

# "Nearby" skill for each skill (used for the 1 nearby future probe).
NEARBY = {
    "fact_verification": "multihop",
    "multihop": "fact_verification",
    "quantitative": "tool_use",
    "tool_use": "quantitative",
}

# "Unrelated" skill for each skill (used for the 1 unrelated future probe).
UNRELATED = {
    "fact_verification": "quantitative",
    "multihop": "tool_use",
    "quantitative": "fact_verification",
    "tool_use": "multihop",
}

# The four persistent subagents and their declared specialisation.
AGENTS = {
    "verifier": "fact_verification",
    "multihop_reasoner": "multihop",
    "quant_reasoner": "quantitative",
    "tool_specialist": "tool_use",
}

CAUSES = ["missing_context", "tool_failure", "out_of_specialty", "persistent_reasoning"]
TRANSIENT_CAUSES = {"missing_context", "tool_failure"}

CONDITIONS = [
    "no_memory",            # 1
    "raw_history",          # 2
    "outcome_only",         # 3
    "skill_conditioned",    # 4
    "claimgraph",           # 5
    "claimgraph_update",    # 6
]

ABLATIONS = [
    "no_parent_links",
    "no_tool_state",
    "no_cause_label",
    "no_persistence_resolution",
    "no_update_rule",
]


@dataclass
class ExperimentConfig:
    # ---- data ----
    tasks_per_source: int = 50
    probe_pool_per_source: int = 200      # held-out pool used for future probes
    calibration_per_skill: int = 25       # Phase A tasks per (agent, skill)
    split: tuple = (0.6, 0.2, 0.2)
    data_seed: int = 13

    # ---- design ----
    conditions: list = field(default_factory=lambda: list(CONDITIONS))
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    probes_affected: int = 2
    probes_nearby: int = 1
    probes_unrelated: int = 1
    failure_mode: str = "force"           # "force" (fixed r-) or "induce" (keep only natural r-)

    # ---- controlled persistent-weakness profile (option (a) in the proposal) ----
    persistent_drop_affected: float = 0.30
    persistent_drop_nearby: float = 0.10

    # ---- utility ----
    verify_cost: float = 0.15             # utility cost of one verification
    verify_catch_rate: float = 0.85       # P(verification catches+fixes an error)
    call_cost: float = 0.0

    # ---- backend ----
    backend: str = "mock"                 # "mock" | "openai" (any OpenAI-compatible server, e.g. vLLM)
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_concurrency: int = 64
    cache_path: str = "runs/llm_cache.sqlite"

    # ---- output-token budgets per call type (also used by the estimator) ----
    max_tokens: dict = field(default_factory=lambda: {
        "subagent": 400,
        "verify": 250,
        "attribute_trust": 450,
        "delegate": 150,
        "probe_verify": 250,
    })

    out_dir: str = "runs/default"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=list)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw = json.load(f)
        if "split" in raw:
            raw["split"] = tuple(raw["split"])
        return cls(**raw)
