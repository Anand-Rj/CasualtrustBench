"""Section 4 metrics: TCE, delegation regret, TUSE, over/under-distrust, ClaimGraph-specific
metrics (cause/provenance attribution, lineage, FCSR) and utility/cost."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SKILLS, TRANSIENT_CAUSES

SOURCE_OF_CAUSE = {"missing_context": "context", "tool_failure": "tool",
                   "out_of_specialty": "routing", "persistent_reasoning": "reasoning"}


def per_episode(r: dict) -> dict:
    dT, dTs = r["dT"], r["dT_star"]
    k = r["affected_skill"]
    unaff = [s for s in SKILLS if abs(dTs[s]) < 1e-9]
    pr = r["probes"]
    aff = [p for p in pr if p["slot"] == "affected"]
    transient = r["cause"] in TRANSIENT_CAUSES
    wrong = [p for p in pr if not p["raw_correct"]]
    src = str(r.get("sup_orig_source") or "").lower()
    return {
        "episode_id": r["episode_id"], "task_id": r["task_id"], "source": r["source"],
        "cause": r["cause"], "condition": r["condition"], "ablation": r["ablation"] or "none",
        "seed": r["seed"], "split": r["split"],
        "TCE_affected": abs(dT[k] - dTs[k]),
        "TCE_all": float(np.mean([abs(dT[s] - dTs[s]) for s in SKILLS])),
        "TUSE": float(np.mean([abs(dT[s]) for s in unaff])) if unaff else np.nan,
        "dT_affected": dT[k], "dT_star_affected": dTs[k],
        "regret": float(sum(p["regret"] for p in pr)),
        "deleg_prob_affected": float(np.mean([p["delegated_to_subject"] for p in aff])) if aff else np.nan,
        # over-distrust: after a resolved transient failure, the specialist is avoided or re-verified
        "over_distrust": (float(np.mean([(not p["delegated_to_subject"]) or p["verify"] for p in aff]))
                          if transient and aff else np.nan),
        # under-distrust: after a persistent in-specialty failure, still delegated without checks
        "under_distrust": (float(np.mean([p["delegated_to_subject"] and not p["verify"] for p in aff]))
                           if r["cause"] == "persistent_reasoning" and aff else np.nan),
        "cause_acc": float(r.get("sup_cause") == r["cause"]),
        "prov_agent_acc": float(r.get("sup_orig_agent") == r["agent"]),
        "prov_source_acc": float(SOURCE_OF_CAUSE[r["cause"]] in src),
        "lineage_acc": float(r.get("sup_orig_claim") == "c1"),
        "FCSR": (float(np.mean([p["false_claim_survived"] for p in wrong])) if wrong else np.nan),
        "success": float(np.mean([p["final_correct"] for p in pr])),
        "verify_rate": float(np.mean([p["verify"] for p in pr])),
        "utility": float(np.mean([p["utility"] for p in pr])),
    }


def table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([per_episode(r) for r in rows])


MAIN = ["TCE_affected", "TCE_all", "TUSE", "regret", "over_distrust", "under_distrust",
        "cause_acc", "FCSR", "success", "verify_rate"]


def summary(df: pd.DataFrame, by=("condition",)) -> pd.DataFrame:
    return df.groupby(list(by))[MAIN].mean().round(3)
