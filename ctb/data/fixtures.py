"""Synthetic offline tasks with realistic lengths.

Used for unit tests and for the dry run that measures prompt sizes when the real
datasets cannot be downloaded. Lengths approximate the real sources:
FEVER claim + ~6 evidence sentences; HotpotQA distractor = 10 paragraphs x ~4 sentences;
GSM8K = 4-6 sentence word problem. Content is generated, not copied from the datasets.
"""
from __future__ import annotations

import random

from ..config import SOURCE_SKILL
from . import dbtool
from .schema import Evidence, Task

_WORDS = ("the river valley council museum station archive league festival province "
          "engineer novelist composer founded located directed published annual northern "
          "former historic regional national district university century album series "
          "film company railway bridge island harbor").split()
_NAMES = ["Arden Hollow", "Marlow Crest", "Velia Stone", "Tobin Reach", "Calder Fenn",
          "Isra Valen", "Odessa Pike", "Bram Ellery", "Corvin Hale", "Lyra Mendez",
          "Kestrel Bay", "Nadir Okafor", "Sable Point", "Wren Abara"]


def _sentence(rng, n=22):
    w = [rng.choice(_WORDS) for _ in range(n)]
    w[0] = w[0].capitalize()
    return " ".join(w) + "."


def _fever(i, rng):
    gold = [Evidence("e0", rng.choice(_NAMES), _sentence(rng, 26)),
            Evidence("e1", rng.choice(_NAMES), _sentence(rng, 24))]
    dis = [Evidence(f"d{k}", "Retrieved passage", _sentence(rng, 24)) for k in range(4)]
    ev = gold + dis
    rng.shuffle(ev)
    label = rng.choice(["SUPPORTS", "REFUTES"])
    claim = f"{rng.choice(_NAMES)} was " + " ".join(rng.choice(_WORDS) for _ in range(10)) + "."
    return Task(f"fever-syn{i}", "fever", SOURCE_SKILL["fever"], claim, ev, label,
                ["e0", "e1"], "REFUTES" if label == "SUPPORTS" else "SUPPORTS", "label")


def _hotpot(i, rng):
    titles = rng.sample(_NAMES, 10)
    ev = []
    for ti, t in enumerate(titles):
        for si in range(rng.randint(3, 5)):
            ev.append(Evidence(f"p{ti}s{si}", t, _sentence(rng, rng.randint(18, 28))))
    q = (f"Which {rng.choice(_WORDS)} associated with {titles[0]} was "
         + " ".join(rng.choice(_WORDS) for _ in range(9)) + "?")
    return Task(f"hotpot-syn{i}", "hotpotqa", SOURCE_SKILL["hotpotqa"], q, ev, titles[1],
                ["p0s0", "p1s0"], titles[2], "short")


def _gsm(i, rng):
    a, b, c = rng.randint(3, 40), rng.randint(2, 9), rng.randint(5, 60)
    sents = [f"{rng.choice(_NAMES)} buys {a} boxes of supplies for a community workshop.",
             f"Each box contains {b} packs, and every pack holds the same number of items.",
             f"Later that week {c} additional items arrive from a regional donor.",
             "Some of the items are set aside for a second event next month."]
    ev = [Evidence(f"s{j}", "Problem statement", s) for j, s in enumerate(sents)]
    gold = a * b + c
    return Task(f"gsm8k-syn{i}", "gsm8k", SOURCE_SKILL["gsm8k"],
                "How many items are available in total?", ev, str(gold), ["s0", "s1", "s2"],
                str(a * b), "number")


def synthetic_tasks(source: str, n: int, seed: int) -> list[Task]:
    rng = random.Random(f"{source}-{seed}")
    if source == "dbtool":
        return dbtool.generate_tasks(n, seed)
    fn = {"fever": _fever, "hotpotqa": _hotpot, "gsm8k": _gsm}[source]
    return [fn(i, rng) for i in range(n)]
