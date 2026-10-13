"""Loaders for the four source datasets named in the proposal.

* FEVER      -> Hugging Face (claims with gold evidence sentences), NEI claims dropped
* HotpotQA   -> Hugging Face `hotpotqa/hotpot_qa`, distractor setting, validation split
* GSM8K      -> Hugging Face `openai/gsm8k`, main config, test split
* Tool tasks -> AgentBench-style DB tasks generated locally on SQLite, so the tool can be
                failed / restored deterministically (see dbtool.py)

Each loader returns `Task` objects that already carry the fixed target wrong answer r-.
If Hugging Face is unreachable, `load_source(..., offline=True)` returns synthetic
fixture tasks of realistic length so the full pipeline can still be dry-run.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ..config import SOURCE_SKILL
from .schema import Evidence, Task
from . import dbtool, fixtures

FEVER_HF_CANDIDATES = [
    ("copenlu/fever_gold_evidence", None, "validation"),
    ("copenlu/fever_gold_evidence", None, "train"),
]
HOTPOT_HF = ("hotpotqa/hotpot_qa", "distractor", "validation")
GSM8K_HF = ("openai/gsm8k", "main", "test")


def _hf(name, config, split):
    from datasets import load_dataset  # imported lazily so offline runs need nothing
    return load_dataset(name, config, split=split)


# --------------------------------------------------------------------------- FEVER
def _fever_from_row(i: int, row: dict, rng: random.Random, distractor_pool: list[str]) -> Task | None:
    label = (row.get("label") or "").upper()
    if label not in {"SUPPORTS", "REFUTES"}:
        return None
    raw_ev = row.get("evidence") or []
    ev: list[Evidence] = []
    for j, item in enumerate(raw_ev):
        # copenlu format: [title, line_id, sentence]
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            ev.append(Evidence(f"e{j}", str(item[0]).replace("_", " "), str(item[2])))
    if not ev:
        return None
    gold_ids = [e.eid for e in ev]
    # add retrieved-but-irrelevant sentences so the subagent sees a realistic evidence set
    for k, sent in enumerate(rng.sample(distractor_pool, min(4, len(distractor_pool)))):
        ev.append(Evidence(f"d{k}", "Retrieved passage", sent))
    rng.shuffle(ev)
    wrong = "REFUTES" if label == "SUPPORTS" else "SUPPORTS"
    return Task(f"fever-{i}", "fever", SOURCE_SKILL["fever"], row["claim"], ev, label,
                gold_ids, wrong, answer_format="label")


def load_fever(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    last_err = None
    for name, cfg, split in FEVER_HF_CANDIDATES:
        try:
            ds = _hf(name, cfg, split)
            break
        except Exception as e:  # pragma: no cover - network dependent
            last_err = e
    else:
        raise RuntimeError(f"Could not load FEVER from HF: {last_err}")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    pool = [ds[i]["evidence"][0][2] for i in idx[:2000] if ds[i].get("evidence")]
    out = []
    for i in idx:
        t = _fever_from_row(i, ds[i], rng, pool)
        if t:
            out.append(t)
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- HotpotQA
def _hotpot_from_row(row: dict, rng: random.Random) -> Task | None:
    titles = row["context"]["title"]
    sents = row["context"]["sentences"]
    ev: list[Evidence] = []
    loc = {}
    for ti, (title, ss) in enumerate(zip(titles, sents)):
        for si, s in enumerate(ss):
            eid = f"p{ti}s{si}"
            ev.append(Evidence(eid, title, s.strip()))
            loc[(title, si)] = eid
    gold_ids = [loc[(t, s)] for t, s in zip(row["supporting_facts"]["title"],
                                           row["supporting_facts"]["sent_id"]) if (t, s) in loc]
    if len(gold_ids) < 2:
        return None
    ans = row["answer"].strip()
    if ans.lower() in {"yes", "no"}:
        wrong = "no" if ans.lower() == "yes" else "yes"
    else:
        cands = [t for t in titles if t.lower() != ans.lower() and t.lower() not in row["question"].lower()]
        if not cands:
            return None
        wrong = rng.choice(cands)
    return Task(f"hotpot-{row['id']}", "hotpotqa", SOURCE_SKILL["hotpotqa"], row["question"], ev,
                ans, gold_ids, wrong, answer_format="short")


def load_hotpot(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    ds = _hf(*HOTPOT_HF)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        t = _hotpot_from_row(ds[i], rng)
        if t:
            out.append(t)
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- GSM8K
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _gsm_from_row(i: int, row: dict, rng: random.Random) -> Task | None:
    sol = row["answer"]
    final = sol.split("####")[-1].strip().replace(",", "")
    # intermediate results from calculator annotations <<a*b=c>>
    inter = [m.split("=")[-1] for m in re.findall(r"<<([^>]+)>>", sol)]
    inter = [x for x in inter if x and x != final]
    wrong = inter[-1] if inter else str(int(float(final)) + rng.choice([-10, -2, 2, 10]))
    wrong = wrong.rstrip(".")
    # split the problem into sentences = the "evidence" items
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", row["question"]) if s.strip()]
    if len(sents) < 3:
        return None
    question = sents[-1]
    ev = [Evidence(f"s{j}", "Problem statement", s) for j, s in enumerate(sents[:-1])]
    numeric = [e.eid for e in ev if _NUM.search(e.text)]
    if not numeric:
        return None
    return Task(f"gsm8k-{i}", "gsm8k", SOURCE_SKILL["gsm8k"], question, ev, final,
                numeric, wrong, answer_format="number")


def load_gsm8k(n: int, seed: int) -> list[Task]:
    rng = random.Random(seed)
    ds = _hf(*GSM8K_HF)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        t = _gsm_from_row(i, ds[i], rng)
        if t:
            out.append(t)
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- dispatcher
LOADERS = {"fever": load_fever, "hotpotqa": load_hotpot, "gsm8k": load_gsm8k,
           "dbtool": dbtool.generate_tasks}


def load_source(source: str, n: int, seed: int, offline: bool = False) -> list[Task]:
    if offline:
        return fixtures.synthetic_tasks(source, n, seed)
    return LOADERS[source](n, seed)


def save_tasks(tasks: list[Task], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict()) + "\n")


def read_tasks(path: str | Path) -> list[Task]:
    with open(path) as f:
        return [Task.from_dict(json.loads(l)) for l in f if l.strip()]
