from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import httpx

from .core import PROTOCOL, canonical, digest

SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "maxLength": 260},
        "choice": {"type": "string", "enum": ["A", "B"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "cites": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    },
    "required": ["message", "choice", "confidence", "cites"],
    "additionalProperties": False,
}


def decision_schema(model):
    schema=copy.deepcopy(SCHEMA)
    if 'citation_max_length' in model:
        schema['properties']['cites']['items']['maxLength']=model['citation_max_length']
    return schema


def apply_output_constraints(payload,model):
    if model.get('structured_grammar_path'):
        payload.pop('response_format',None)
        grammar=Path(model['structured_grammar_path']).read_text(encoding='utf-8')
        if model.get('structured_grammar_sha256') and hashlib.sha256(grammar.encode()).hexdigest()!=model['structured_grammar_sha256']:
            raise RuntimeError('Frozen structured grammar hash mismatch')
        payload['structured_outputs']={'grammar':grammar}
    else:
        payload['response_format']['json_schema']['schema']=decision_schema(model)
    return payload


class LLMClient:
    def __init__(self, model: dict, cache_path: Path, concurrency: int = 32):
        self.model = model
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, request TEXT, response TEXT, parsed TEXT, created REAL, seconds REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS failures (key TEXT, attempt INTEGER, error TEXT, created REAL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS failed_responses (key TEXT, attempt INTEGER, request TEXT, status INTEGER, response TEXT, created REAL)")
        self.db.commit()
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(model.get('request_timeout_seconds',240), connect=20), limits=httpx.Limits(max_connections=concurrency + 4))
        self.sem = asyncio.Semaphore(concurrency)
        self.locks: dict[str, asyncio.Lock] = {}
        self.actual_calls = self.cache_hits = 0
        self.server_info = None

    async def inspect(self):
        response = await self.http.get(self.model["url"] + "/models")
        response.raise_for_status()
        info = response.json()
        actual = [m for m in info["data"] if m["id"] == self.model["id"]]
        if not actual:
            raise RuntimeError(f"Model mismatch: {info}")
        self.server_info = actual[0]
        return actual[0]

    async def close(self):
        await self.http.aclose()
        self.db.commit()
        self.db.close()

    async def complete(self, system: str, prompt: str, seed: int, temperature: float = 0.7, max_tokens: int = 240) -> dict:
        max_tokens = self.model.get('max_completion_tokens',max_tokens)
        payload = {
            "model": self.model["id"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": temperature, "top_p": 0.95, "max_tokens": max_tokens,
            "seed": seed,
            "response_format": {"type": "json_schema", "json_schema": {"name": "decision", "strict": True, "schema": decision_schema(self.model)}},
        }
        if self.model.get("disable_thinking"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        apply_output_constraints(payload,self.model)
        key = digest([PROTOCOL, self.model.get("fingerprint", self.model["id"]), payload])
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self.db.execute("SELECT parsed FROM calls WHERE key=?", (key,)).fetchone()
            if cached:
                self.cache_hits += 1
                return {**json.loads(cached[0]), "call_key": key, "cached": True}
            async with self.sem:
                for attempt in range(4):
                    started = time.time()
                    response = None
                    try:
                        response = await self.http.post(self.model["url"] + "/chat/completions", json=payload)
                        response.raise_for_status()
                        raw = response.json()
                        choice = raw["choices"][0]
                        if choice["finish_reason"] != "stop":
                            raise ValueError(f"Incomplete response: {choice['finish_reason']} {choice['message']}")
                        content = choice["message"]["content"]
                        try:
                            parsed = json.loads(content)
                        except json.JSONDecodeError:
                            # Llama/xgrammar artifact (20261116): the model emits the merged token `"]}` inside an
                            # empty cites list, yielding `"cites": ["]}`. message/choice/confidence are complete;
                            # repair ONLY this exact tail to an empty cite list and log the key in `repairs`.
                            tail = '"cites": ["]}'
                            if not (self.model["id"] in os.environ.get("REPAIR_DANGLING_CITE_MODELS", "").split(",") and content.endswith(tail)):
                                raise
                            parsed = json.loads(content[: -len('["]}')] + "[]}")
                            self.db.execute("CREATE TABLE IF NOT EXISTS repairs(key TEXT, kind TEXT, created REAL)")
                            self.db.execute("INSERT INTO repairs VALUES(?,?,?)", (key, "dangling_cite", time.time()))
                        self.validate(parsed)
                        if 'citation_max_length' in self.model and any(len(c)>self.model['citation_max_length'] for c in parsed['cites']):
                            raise ValueError('Citation exceeds configured maximum length')
                        self.db.execute("INSERT OR REPLACE INTO calls VALUES(?,?,?,?,?,?)", (key, canonical(payload), canonical(raw), canonical(parsed), started, time.time() - started))
                        self.db.commit()
                        self.actual_calls += 1
                        return {**parsed, "call_key": key, "cached": False}
                    except Exception as exc:
                        detail = f'{type(exc).__name__}: {exc}'
                        if isinstance(exc, httpx.HTTPStatusError):
                            detail += " " + exc.response.text[:1500]
                        self.db.execute("INSERT INTO failures VALUES(?,?,?,?)", (key, attempt, detail, time.time()))
                        self.db.execute("INSERT INTO failed_responses VALUES(?,?,?,?,?,?)", (key, attempt, canonical(payload), response.status_code if response is not None else None, response.text if response is not None else None, started))
                        self.db.commit()
                        # Configuration errors are not silently repaired by changing model settings.
                        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500 and exc.response.status_code != 429:
                            raise RuntimeError(detail) from exc
                        if attempt == 3:
                            raise RuntimeError(detail) from exc
                        await asyncio.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    @staticmethod
    def validate(obj):
        if set(obj) != set(SCHEMA["required"]):
            raise ValueError(f"Unexpected keys: {list(obj)}")
        if obj["choice"] not in ("A", "B") or not isinstance(obj["confidence"], (int, float)) or not 0 <= obj["confidence"] <= 1:
            raise ValueError("Invalid decision")
        if not isinstance(obj["message"], str) or len(obj["message"]) > 260:
            raise ValueError("Invalid message")
        if not isinstance(obj["cites"], list) or len(obj["cites"]) > 6 or not all(isinstance(x, str) for x in obj["cites"]):
            raise ValueError("Invalid citations")
