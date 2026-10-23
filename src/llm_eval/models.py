from __future__ import annotations

import re
import time
from collections.abc import Callable

import httpx

from llm_eval.runner import Model, ModelResponse

# --- Deterministic rule-based stand-ins -------------------------------------------------------
# These are NOT language models. They give the harness reproducible behaviour for the demo and
# for tests: v2 is a "refactor" that improves some cases and regresses another.

CAPITALS = {"france": "Paris", "germany": "Berlin"}
CAPITALS_V2 = {**CAPITALS, "japan": "Tokyo"}
MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(),
    start=1)}
POSITIVE = {"love", "great", "good", "excellent", "amazing"}
NEGATIVE = {"terrible", "broken", "bad", "awful", "hate"}


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(round(x, 4))


def _arith(prompt: str, integer_division: bool) -> str | None:
    m = re.search(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)", prompt)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op == "/":
        if b == 0:
            return "undefined"
        return str(a // b) if integer_division else _fmt(a / b)
    return str({"+": a + b, "-": a - b, "*": a * b}[op])


def _date(prompt: str) -> str | None:
    m = re.search(r"([A-Za-z]+) (\d{1,2}), (\d{4})", prompt)
    if not m or m.group(1).lower() not in MONTHS:
        return None
    return f"{m.group(3)}-{MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"


def _person(prompt: str) -> str | None:
    m = re.search(r"([A-Z][a-z]+) is (\d+) years old", prompt)
    return f'{{"name": "{m.group(1)}", "age": {m.group(2)}}}' if m else None


def _sentiment(prompt: str, handle_negation: bool) -> str | None:
    if "sentiment" not in prompt.lower():
        return None
    text = prompt.split(":", 1)[-1].lower()
    words = set(re.findall(r"[a-z]+", text))
    negated = handle_negation and "not" in words
    if words & NEGATIVE:
        return "negative"
    if words & POSITIVE:
        return "negative" if negated else "positive"
    return "negative" if negated else "positive"


def _capital(prompt: str, table: dict[str, str]) -> str | None:
    m = re.search(r"capital of ([A-Za-z]+)", prompt)
    return table.get(m.group(1).lower(), "I don't know") if m else None


def echo(prompt: str) -> str:
    return prompt


def rules_v1(prompt: str) -> str:
    for result in (_arith(prompt, False), _person(prompt), _sentiment(prompt, False),
                   _capital(prompt, CAPITALS), _date(prompt)):
        if result is not None:
            return result
    return "I don't know"


def rules_v2(prompt: str) -> str:
    for result in (_arith(prompt, True), _person(prompt), _sentiment(prompt, True),
                   _capital(prompt, CAPITALS_V2), _date(prompt)):
        if result is not None:
            return result
    return "I don't know"


BASELINES: dict[str, Model] = {"echo": echo, "rules-v1": rules_v1, "rules-v2": rules_v2}


# --- Real model adapter ----------------------------------------------------------------------

RETRYABLE = {429, 500, 502, 503, 504}
MAX_WAIT = 90.0  # a longer wait means a daily quota, not a rate spike: fail instead of hanging


class OpenAICompatModel:
    """Calls any OpenAI-compatible /chat/completions endpoint (OpenAI, Ollama, vLLM, LM Studio).

    Sampling parameters are sent with each request and exposed via `config` so runs record exactly
    how they were produced.
    """

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None, system: str | None = None,
                 temperature: float = 0.0, top_p: float | None = None, seed: int | None = None,
                 max_tokens: int | None = None, client: httpx.Client | None = None,
                 timeout: float = 120.0, retries: int = 5) -> None:
        self.retries = retries
        self.model, self.system = model, system
        self.temperature, self.top_p, self.seed, self.max_tokens = temperature, top_p, seed, max_tokens
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=timeout)

    @property
    def config(self) -> dict:
        return {"type": "openai-compatible", "model": self.model, "base_url": self.base_url,
                "temperature": self.temperature, "top_p": self.top_p, "seed": self.seed,
                "max_tokens": self.max_tokens, "system_prompt": self.system}

    def __call__(self, prompt: str) -> ModelResponse:
        messages = ([{"role": "system", "content": self.system}] if self.system else [])
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages, "temperature": self.temperature}
        for key, value in (("top_p", self.top_p), ("seed", self.seed), ("max_tokens", self.max_tokens)):
            if value is not None:
                body[key] = value
        resp = self.client.post(self.url, headers=self.headers, json=body)
        for attempt in range(self.retries):
            if resp.status_code not in RETRYABLE:
                break
            # Hosted free tiers rate-limit per minute: honour Retry-After, else back off.
            hinted = resp.headers.get("retry-after") or (
                m[1] if (m := re.search(r'"retryDelay":\s*"([\d.]+)s"', resp.text)) else None)
            try:
                delay = float(hinted)
            except (TypeError, ValueError):
                delay = min(5 * 2 ** attempt, MAX_WAIT)
            if delay > MAX_WAIT:
                break
            time.sleep(delay)
            resp = self.client.post(self.url, headers=self.headers, json=body)
        if resp.is_error:
            raise httpx.HTTPStatusError(f"{resp.status_code} from {self.url}: {resp.text[:300]}",
                                        request=resp.request, response=resp)
        data = resp.json()
        usage = data.get("usage") or {}
        return ModelResponse(
            text=data["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


def resolve_model(spec: str, api_key: str | None = None, **params) -> Callable[[str], object]:
    """'rules-v1' | 'openai:MODEL@BASE_URL' (BASE_URL optional). params: system, temperature,
    top_p, seed, max_tokens (only used for remote models)."""
    if spec in BASELINES:
        return BASELINES[spec]
    if spec.startswith("openai:"):
        model, _, base = spec[len("openai:"):].partition("@")
        return OpenAICompatModel(model, base or "https://api.openai.com/v1", api_key, **params)
    raise ValueError(f"Unknown model {spec!r}. Baselines: {sorted(BASELINES)} or openai:MODEL@URL")
