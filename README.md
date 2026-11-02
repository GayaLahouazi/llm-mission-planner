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

Run on a free Colab T4 GPU, **30 random missions** (seeds 0–29, same generator settings), greedy decoding, at most
160 new tokens, up to 3 repairs. Qwen2.5-3B-Instruct (fp16) runs with `notebooks/colab_llm_benchmark.ipynb`;
Qwen2.5-7B-Instruct was loaded in 4-bit (NF4, bitsandbytes) to fit the GPU and plugged into the same benchmark code.
The classical planners were re-run on the same 30 missions for comparison.

| Planner | Valid plans | Mean gap to optimal (valid plans) | Mean time / mission |
|---|---:|---:|---:|
| Nearest neighbour | 13 % (4/30) | +7.5 % | – |
| Earliest deadline first | 40 % (12/30) | +8.7 % | – |
| Exact branch & bound | 100 % | 0 % (reference) | – |
| Qwen2.5-3B – LLM alone | 0 % | – | 2.7 s |
| Qwen2.5-3B – LLM + RAG | 0 % | – | 3.7 s |
| Qwen2.5-3B – LLM + RAG + Z3 repair (≤ 3) | 0 % | – | 15.9 s |
| Qwen2.5-7B (4-bit) – LLM alone | 10 % (3/30) | +19.1 % | 2.5 s |
| Qwen2.5-7B (4-bit) – LLM + RAG | 7 % (2/30) | +24.9 % | 2.9 s |
| Qwen2.5-7B (4-bit) – LLM + RAG + Z3 repair (≤ 3) | 13 % (4/30) | +16.2 % | 11.8 s |

Z3 verification took 3.7–4.0 ms per plan in these runs. With the 7B model, 2/30 first answers were valid and 4/30
after repairs (1.25 repairs on average for the plans that ended valid).

What this shows:

* **The verifier is what makes the system safe:** every invalid LLM plan was caught by Z3 and turned into readable
  reasons; none was returned as valid.
* **Small open LLMs cannot compute a constrained route.** They lose to a simple rule (earliest deadline first, 40 %)
  and to the exact solver. Typical failure (3B, mission 0): the model lists the waypoints in index order (W1 … W7) and,
  when told "route length 55.6 km exceeds the usable range 51.0 km", only moves `depot` in the list, so the route
  does not change.
* **Repair feedback helps a bigger model** (7B: 2 → 4 valid plans) but not a small one: model size and feedback
  quality both matter.
* **Consequence for the design:** the LLM's place is to read the brief and the procedures and to *orchestrate*
  solvers (choose and parameterise the exact planner, explain the result), not to replace them. This matches the
  LLM-Modulo argument of Kambhampati et al.

Caveats: 30 missions only, so differences of one or two missions (e.g. 7B alone vs + RAG) are within noise;
4-bit quantisation may cost the 7B some accuracy.

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

An LLM that calls the exact planner as a tool (choosing and parameterising it) instead of writing the route itself,
richer repair feedback (which leg makes the route too long), larger models, several drones (assignment + routing),
uncertain travel times (chance constraints), an exact MILP formulation to compare with branch & bound, and
fine-tuning a small LLM on solver-generated plans.

## References

* de Moura & Bjørner, *Z3: An Efficient SMT Solver*, TACAS 2008.
* Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009.
* Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020.
* Kambhampati et al., *LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks*, ICML 2024.

---
Author: **Gaya Lahouazi**, M2 AI2D (Sorbonne Université). Links with the MADI course (decision) and with
industrial work on LLM-based mission planners coupled with formal tools.
