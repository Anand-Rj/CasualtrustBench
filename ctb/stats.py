"""Statistical analysis plan: paired bootstrap CIs at the base-task level, mixed-effects
models, and paired McNemar tests for binary decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_bootstrap(df: pd.DataFrame, metric: str, a: str, b: str, n_boot: int = 5000,
                     seed: int = 0) -> dict:
    """Mean difference (b - a) in `metric`, paired by base task, with 95% bootstrap CI."""
    g = df.groupby(["task_id", "condition"])[metric].mean().unstack()
    g = g[[a, b]].dropna()
    diff = (g[b] - g[a]).to_numpy()
    if len(diff) == 0:
        return {"metric": metric, "a": a, "b": b, "n": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, (n_boot, len(diff)), replace=True).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"metric": metric, "a": a, "b": b, "n_tasks": len(diff), "mean_diff": float(diff.mean()),
            "ci95": [float(lo), float(hi)], "p_boot_two_sided": float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))}


def mixed_effects(df: pd.DataFrame, metric: str, ref: str = "outcome_only"):
    """metric ~ condition * cause + (1 | task). Add `model` as a second grouping factor in
    the robustness study via vc_formula."""
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[metric]).copy()
    d["condition"] = pd.Categorical(d["condition"])
    formula = f"{metric} ~ C(condition, Treatment('{ref}')) * C(cause)"
    vc = {"model": "0 + C(model)"} if "model" in d and d["model"].nunique() > 1 else None
    m = smf.mixedlm(formula, d, groups=d["task_id"], vc_formula=vc)
    return m.fit(method="lbfgs", reml=True)


def mcnemar(df_probe: pd.DataFrame, col: str, a: str, b: str) -> dict:
    """Paired McNemar on a binary per-probe decision between conditions a and b."""
    from statsmodels.stats.contingency_tables import mcnemar as _mc
    piv = df_probe.pivot_table(index=["episode_id", "seed", "probe_idx"], columns="condition",
                               values=col, aggfunc="first")[[a, b]].dropna().astype(bool)
    tab = pd.crosstab(piv[a], piv[b]).reindex(index=[False, True], columns=[False, True], fill_value=0)
    r = _mc(tab.to_numpy(), exact=tab.to_numpy().sum() < 50)
    return {"col": col, "a": a, "b": b, "table": tab.to_numpy().tolist(),
            "statistic": float(r.statistic), "p": float(r.pvalue)}


def probe_frame(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        for i, p in enumerate(r["probes"]):
            out.append({"episode_id": r["episode_id"], "seed": r["seed"], "condition": r["condition"],
                        "cause": r["cause"], "probe_idx": i, **p})
    return pd.DataFrame(out)
