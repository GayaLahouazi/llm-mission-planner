"""LLM back-ends and the propose -> verify -> repair loop.

Back-ends (all expose `chat(messages) -> str`):
    OpenAICompatible : any server speaking the OpenAI chat API
                       (Ollama, vLLM, llama.cpp server, Groq, Mistral, OpenAI...)
    HFTransformers   : a local Hugging Face model (used in the Colab notebook)
    Scripted         : returns pre-written answers (unit tests)
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from .rag import retrieve_rules
from .scenario import Scenario, brief
from .verifier import Verdict, verify

SYSTEM = ("You are a mission planner for an inspection drone. You return ONLY a JSON object "
          'of the form {"order": ["W1", "W2", ...]} with every waypoint exactly once, in visiting order.')


class OpenAICompatible:
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", api_key_env: str = "LLM_API_KEY",
                 temperature: float = 0.0):
        import requests  # imported lazily: not needed for the tests
        self.requests, self.model, self.url = requests, model, base_url.rstrip("/") + "/chat/completions"
        self.key, self.temperature = os.environ.get(api_key_env, "none"), temperature

    def chat(self, messages: list[dict]) -> str:
        r = self.requests.post(self.url, timeout=300, headers={"Authorization": f"Bearer {self.key}"},
                               json={"model": self.model, "messages": messages, "temperature": self.temperature})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class HFTransformers:
    def __init__(self, model: str = "Qwen/Qwen2.5-3B-Instruct", max_new_tokens: int = 160):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model=model, device_map="auto", torch_dtype="auto")
        self.max_new_tokens = max_new_tokens

    def chat(self, messages: list[dict]) -> str:
        out = self.pipe(messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"]


class Scripted:
    def __init__(self, answers: list[str]):
        self.answers = list(answers)

    def chat(self, messages):
        return self.answers.pop(0)


def parse_order(text: str, drop_depot: bool = True) -> list[str] | None:
    """Extract the JSON plan from a free-form answer (models sometimes add prose or ``` fences).

    The depot is implicit (procedures DRN-01 and DRN-10). Small models often add it at the start or the end of the
    route anyway; with `drop_depot` (default) those mentions are removed before verification, so that the benchmark
    measures planning rather than formatting. `drop_depot=False` gives the strict behaviour."""
    m = re.search(r"\{.*\}", text, re.S)
    order = None
    if m:
        try:
            raw = json.loads(m.group(0)).get("order")
            if isinstance(raw, list):
                order = [str(o).strip() for o in raw]
        except (json.JSONDecodeError, AttributeError):
            pass
    if order is None:
        order = re.findall(r"\bW\d+\b", text)       # fallback: list of waypoint names in the text
    if drop_depot:
        order = [o for o in order if o.lower() != "depot"]
    return order or None


@dataclass
class PlanResult:
    order: list[str] | None
    verdict: Verdict | None
    iterations: int
    seconds: float
    first_valid: bool
    retrieved: list[str] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)


def plan(s: Scenario, llm, use_rag: bool = True, max_repairs: int = 3, drop_depot: bool = True) -> PlanResult:
    """1 proposal + up to `max_repairs` corrections guided by the verifier's feedback."""
    t0 = time.time()
    text = brief(s)
    ids, rules = retrieve_rules(text) if use_rag else ([], "")
    user = text + (f"\n\nOperating procedures that apply:\n{rules}" if rules else "") + \
        "\n\nPropose the shortest route that satisfies every constraint. Answer with JSON only."
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    order, verdict, first_valid = None, None, False
    for it in range(max_repairs + 1):
        answer = llm.chat(messages)
        messages.append({"role": "assistant", "content": answer})
        order = parse_order(answer, drop_depot)
        verdict = verify(s, order) if order else Verdict(False, ["answer is not a valid JSON plan"])
        if it == 0:
            first_valid = verdict.valid
        if verdict.valid:
            return PlanResult(order, verdict, it, time.time() - t0, first_valid, ids, messages)
        messages.append({"role": "user", "content": verdict.feedback() +
                         "\nFix the plan. Answer with the corrected JSON only."})
    return PlanResult(order, verdict, max_repairs, time.time() - t0, first_valid, ids, messages)
