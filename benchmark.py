"""Benchmark of planners on random feasible missions.

Classical planners only (runs anywhere, ~1 min):
    python benchmark.py
With an LLM (OpenAI-compatible server, e.g. Ollama):
    python benchmark.py --llm openai --model qwen2.5:7b --base-url http://localhost:11434/v1
With a Hugging Face model (see notebooks/colab_llm_benchmark.ipynb, free T4 GPU):
    python benchmark.py --llm hf --model Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from planner import llm as llm_mod
from planner.rag import BM25, load_corpus, retrieve_rules
from planner.scenario import brief
from planner.solver import deadline_first, generate_feasible, nearest_neighbour
from planner.verifier import verify

KEY_RULES = ["DRN-02_time_windows", "DRN-03_battery_reserve", "DRN-04_deadline", "DRN-05_no_fly_zones"]


def summarise(rows):
    valid = [r for r in rows if r["valid"]]
    gaps = [r["distance"] / r["optimal"] - 1 for r in valid]
    out = {"valid_rate": len(valid) / len(rows),
           "mean_gap_to_optimal": float(np.mean(gaps)) if gaps else None,
           "optimal_found_rate": float(np.mean([g < 1e-6 for g in gaps] + [False] * (len(rows) - len(valid))))}
    if rows and "iterations" in rows[0]:
        out["first_answer_valid_rate"] = float(np.mean([r["first_valid"] for r in rows]))
        out["mean_repairs_when_valid"] = float(np.mean([r["iterations"] for r in valid])) if valid else None
        out["mean_seconds"] = float(np.mean([r["seconds"] for r in rows]))
    return out


def main(a):
    scenarios = [generate_feasible(seed, n_waypoints=a.waypoints) for seed in range(a.n)]
    results = {"n_scenarios": a.n, "waypoints": a.waypoints, "planners": {}}

    # ---- classical planners + verifier timing -------------------------------------------
    t_verify = []
    for name, fn in {"nearest neighbour": nearest_neighbour, "deadline first": deadline_first,
                     "exact branch & bound (optimal)": None}.items():
        rows = []
        for s, opt_order, opt_d in scenarios:
            order = opt_order if fn is None else fn(s)
            t0 = time.time()
            v = verify(s, order)
            t_verify.append(time.time() - t0)
            rows.append({"valid": v.valid, "distance": v.distance, "optimal": opt_d})
        results["planners"][name] = summarise(rows)
    results["z3_verify_ms_mean"] = 1000 * float(np.mean(t_verify))

    # ---- retrieval: whole-brief query vs query decomposition --------------------------------
    corpus = load_corpus()
    bm25 = BM25(corpus)
    naive, decomp = [], []
    for s, _, _ in scenarios:
        b = brief(s)
        top = bm25.search(b, 5)
        ids, _ = retrieve_rules(b, k=5, corpus=corpus)
        naive.append(np.mean([r in top for r in KEY_RULES]))
        decomp.append(np.mean([r in ids for r in KEY_RULES]))
    results["retrieval_recall_at_5"] = {"whole brief as query": float(np.mean(naive)),
                                        "query decomposition": float(np.mean(decomp))}

    # ---- LLM planner ---------------------------------------------------------------------------
    if a.llm:
        model = llm_mod.HFTransformers(a.model) if a.llm == "hf" else \
            llm_mod.OpenAICompatible(a.model, base_url=a.base_url)
        for label, rag, repairs in (("LLM alone", False, 0), ("LLM + RAG", True, 0),
                                    ("LLM + RAG + Z3 repair loop (<=3)", True, 3)):
            rows = []
            for i, (s, _, opt_d) in enumerate(scenarios):
                r = llm_mod.plan(s, model, use_rag=rag, max_repairs=repairs, drop_depot=not a.strict_format)
                rows.append({"valid": r.verdict.valid, "distance": r.verdict.distance, "optimal": opt_d,
                             "iterations": r.iterations, "first_valid": r.first_valid, "seconds": r.seconds})
                print(f"{label:35s} scenario {i:3d} valid={r.verdict.valid} repairs={r.iterations}", flush=True)
            results["planners"][f"{label} [{a.model}]"] = summarise(rows)

    out = Path("results")
    out.mkdir(exist_ok=True)
    tag = "classical" if not a.llm else a.model.replace("/", "_").replace(":", "_") + ("_strict" if a.strict_format else "")
    (out / f"benchmark_{tag}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--waypoints", type=int, default=7)
    p.add_argument("--llm", choices=["openai", "hf"], default=None)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--strict-format", action="store_true",
                   help="reject plans that mention the (implicit) depot instead of dropping it")
    main(p.parse_args())
