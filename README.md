# Neuro-symbolic mission planning: LLM proposes, Z3 verifies

![tests](https://github.com/GayaLahouazi/llm-mission-planner/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

**Problem.** An inspection drone must visit a set of waypoints and come back to its depot while respecting
time windows, a battery budget, permanent and *temporary* no-fly zones and a mission deadline. Some rules
(battery reserve, safety margin around zones) are not in the mission brief: they are written in the
operator's procedures.

**Idea.** A large language model is good at reading a brief and proposing a plan, but it gives no guarantee.
A formal solver gives guarantees but needs a precise model. This project couples the two:

```
mission brief ──► RAG: retrieve the applicable procedures (BM25, from scratch)
                     │
                     ▼
               LLM proposes a plan  {"order": ["W3", "W1", ...]}
                     │
                     ▼
               Z3 SMT verifier ── valid ──► plan + schedule (with a proof that every constraint holds)
                     │
                  invalid: unsat core → readable reasons
                     │   "W3 must be inspected by t = 45 min", "leg W2 -> W5 crosses zone Z2 while active"
                     └──► sent back to the LLM, which repairs its plan (up to 3 times)
```

The verifier never trusts the LLM: **a plan is only returned if Z3 proves it satisfies every constraint.**

## What is implemented

| Module | Content |
|---|---|
| `planner/scenario.py` | mission model (waypoints, time windows, zones), random generator, natural-language brief |
| `planner/verifier.py` | **Z3 encoding**: schedule variables, waiting allowed, disjunctive constraints for temporary zones, tracked constraints → unsat core → feedback sentences |
| `planner/solver.py` | exact branch & bound planner (optimal reference), greedy heuristics, independent schedule simulator |
| `planner/rag.py` | **BM25 written from scratch** + query decomposition over `docs/procedures/` |
| `planner/llm.py` | LLM back-ends (any OpenAI-compatible server: Ollama, vLLM, Groq, Mistral…; or Hugging Face `transformers`) and the propose → verify → repair loop |
| `benchmark.py` | evaluation on random feasible missions |
| `tests/` | 11 unit tests, including a cross-check: the Z3 verifier and the hand-written simulator must agree on every plan |

## Results

Benchmark: **50 random missions**, 7 waypoints, 2 no-fly zones (one temporary), 40 % of waypoints with a time window.
Every mission is guaranteed feasible (the exact planner found a valid route).

### Classical planners and verifier

| Planner | Valid plans | Mean gap to optimal distance (valid plans) |
|---|---:|---:|
| Nearest neighbour (greedy) | 16 % | +7.7 % |
| Earliest deadline first | 34 % | +8.1 % |
| Exact branch & bound | 100 % | 0 % (reference) |

* Z3 verifies a plan in **3.8 ms** on average (CPU), so it can sit inside an interactive loop.
* The greedy heuristics fail on most missions: time windows and temporary zones make the problem
  much harder than a plain travelling-salesman tour. This is the gap the LLM + verifier loop targets.

### Retrieval (RAG)

Recall@5 of the 4 procedures that really constrain the plan (time windows, battery reserve, deadline, no-fly margin):

| Query strategy | Recall@5 |
|---|---:|
| Whole brief as one BM25 query | 75 % (the battery-reserve rule is never retrieved) |
| **Query decomposition** (one query per clause, union of the best hits) | **100 %** |

Why: the brief repeats words like *service*, *time window* on every waypoint line, so a single query is dominated
by them and the battery rule (mentioned once) is drowned. Splitting the brief into clauses fixes it.

### LLM planner

Run `notebooks/colab_llm_benchmark.ipynb` on a free Colab T4 GPU (Qwen2.5-3B-Instruct, open weights).
It compares *LLM alone*, *LLM + RAG* and *LLM + RAG + Z3 repair loop* on the same missions and reports valid-plan rate,
number of repairs and gap to optimal. **Results will be added here after the run.**

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q tests            # 11 tests, < 1 s
python benchmark.py                  # classical planners + retrieval, ~1 min on CPU

# with a local LLM served by Ollama (https://ollama.com)
ollama pull qwen2.5:7b
python benchmark.py --llm openai --model qwen2.5:7b --base-url http://localhost:11434/v1
```

## Design choices

* **Why Z3 and not hand-written checks?** Temporary zones create disjunctions ("cross before it opens *or*
  after it closes"), and the schedule (waiting times) is a free variable. SMT solves this exactly, and the
  **unsat core** names the constraints that conflict, which is precisely the feedback an LLM needs.
* **Two independent implementations** (Z3 model and a pure-Python earliest-time simulator) are checked against
  each other in the tests: a bug in one of them would show up as a disagreement.
* **No trust in the LLM output:** parsing tolerates prose and code fences, unknown or duplicated waypoints are
  reported, and nothing reaches the user without a SAT answer from the verifier.

## Possible extensions

Several drones (assignment + routing), uncertain travel times (chance constraints), an exact MILP formulation to
compare with branch & bound, and fine-tuning a small LLM on solver-generated plans.

## References

* de Moura & Bjørner, *Z3: An Efficient SMT Solver*, TACAS 2008.
* Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009.
* Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020.
* Kambhampati et al., *LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks*, ICML 2024.

---
Author: **Gaya Lahouazi**, M2 AI2D (Sorbonne Université). Links with the MADI course (decision) and with
industrial work on LLM-based mission planners coupled with formal tools.
