"""Retrieval-Augmented Generation: BM25 retrieval over the procedure documents.

Some rules (battery reserve, safety margin around zones) are NOT in the mission
brief: they only exist in the operator's procedures (docs/procedures). The
retriever finds the relevant procedures and they are pasted into the prompt.

BM25 (Robertson & Zaragoza, 2009) is implemented from scratch, it is ~30 lines.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

STOP = set("a an the of to and or in on at for is are be by with as it its this that from must "
           "never each every all any not no into than then their there these those will can may".split())


def tokenize(text: str) -> list[str]:
    """Lower-case words; pure numbers (coordinates, times) are dropped: they match noise."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and not t.isdigit()]


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.ids = list(docs)
        self.toks = [tokenize(docs[i]) for i in self.ids]
        self.tf = [Counter(t) for t in self.toks]
        self.avgdl = sum(map(len, self.toks)) / len(self.toks)
        n = len(self.toks)
        df = Counter(w for t in self.toks for w in set(t))
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b, self.docs = k1, b, docs

    def scores(self, query: str) -> dict[str, float]:
        q = set(tokenize(query))        # each query term counted once, not once per waypoint line
        out = {}
        for i, tf, toks in zip(self.ids, self.tf, self.toks):
            norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl)
            out[i] = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in q if w in tf)
        return out

    def search(self, query: str, k: int = 4) -> list[str]:
        s = self.scores(query)
        return sorted(s, key=s.get, reverse=True)[:k]


def load_corpus(folder: str | Path = Path(__file__).resolve().parent.parent / "docs" / "procedures"):
    return {p.stem: p.read_text() for p in sorted(Path(folder).glob("*.md"))}


def retrieve_rules(brief: str, k: int = 5, corpus: dict[str, str] | None = None) -> tuple[list[str], str]:
    """Query decomposition: a mission brief mixes several topics (battery, zones,
    time windows...). Searching with the whole brief lets the most repeated topic
    win, so we split it into clauses, keep the best document of each clause, and
    rank the union by its best clause score."""
    corpus = corpus or load_corpus()
    bm25 = BM25(corpus)
    best: dict[str, float] = {}
    for clause in re.split(r"[\n.,;]", brief):
        if len(set(tokenize(clause))) < 2:
            continue
        scores = bm25.scores(clause)
        top = max(scores, key=scores.get)
        if scores[top] > 0:
            best[top] = max(best.get(top, 0.0), scores[top])
    ids = sorted(best, key=best.get, reverse=True)[:k]
    return ids, "\n\n".join(corpus[i].strip() for i in ids)
