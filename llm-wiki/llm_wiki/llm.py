import json
import os
import random
import re
import time
from typing import Optional

import requests

from .types import LLMTool

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLMWIKI_PROVIDER = os.environ.get("LLMWIKI_PROVIDER", "anthropic")
LLMWIKI_MODEL = os.environ.get("LLMWIKI_MODEL", "claude-sonnet-4-20250514")
LLMWIKI_BASE_URL = os.environ.get("LLMWIKI_BASE_URL", "")
MAX_TOKENS = 4096
RETRY_COUNT = 3
RETRY_DELAY = 2.0
RETRY_BASE_MS = 1000
RETRY_MULTIPLIER = 4

LLMWIKI_EMBEDDING_MODEL = os.environ.get("LLMWIKI_EMBEDDING_MODEL", "")
LLMWIKI_EMBEDDING_BASE_URL = os.environ.get("LLMWIKI_EMBEDDING_BASE_URL", "")


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class LLMError(Exception):
    pass


class BaseProvider:
    def complete(self, system: str, prompt: str) -> str:
        raise NotImplementedError

    def extract_concepts(self, system: str, prompt: str) -> Optional[list[dict]]:
        raise NotImplementedError

    def select_pages(self, system: str, prompt: str, max_items: int = 5) -> Optional[list[str]]:
        raise NotImplementedError("Page selection not supported by this provider")

    def embed(self, text: str, input_type: str = "document") -> list[float]:
        raise NotImplementedError("Embedding not supported by this provider")


class AnthropicProvider(BaseProvider):
    def __init__(self):
        self.api_key = ANTHROPIC_API_KEY
        self.model = LLMWIKI_MODEL
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY not set")

    def _retry_call(self, fn):
        last_ex = None
        for attempt in range(RETRY_COUNT):
            try:
                return fn()
            except LLMError as e:
                last_ex = e
                resp_code = getattr(e, "resp_code", 0)
                if 400 <= resp_code < 500 and resp_code != 429:
                    raise
                if attempt < RETRY_COUNT - 1:
                    delay = random.uniform(RETRY_BASE_MS / 2000, RETRY_BASE_MS / 1000) * (RETRY_MULTIPLIER ** attempt)
                    time.sleep(delay)
        raise last_ex

    def _request(self, body: dict) -> dict:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
            timeout=180,
        )
        if resp.status_code != 200:
            e = LLMError(f"Anthropic API error {resp.status_code}: {resp.text}")
            e.resp_code = resp.status_code
            raise e
        return resp.json()

    def complete(self, system: str, prompt: str) -> str:
        body = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        return self._retry_call(lambda: self._complete(body))

    def _complete(self, body: dict) -> str:
        data = self._request(body)
        return data["content"][0]["text"]

    def select_pages(self, system: str, prompt: str, max_items: int = 5) -> Optional[list[str]]:
        return self._retry_call(lambda: self._select_pages(system, prompt, max_items))

    def _select_pages(self, system: str, prompt: str, max_items: int) -> Optional[list[str]]:
        tool_def = {
            "name": "select_pages",
            "description": "Select relevant wiki pages for a question",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": max_items,
                    },
                    "reasoning": {
                        "type": "string",
                    },
                },
                "required": ["pages", "reasoning"],
            },
        }
        body = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [tool_def],
            "tool_choice": {"type": "tool", "name": "select_pages"},
        }
        data = self._request(body)
        for block in data.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "select_pages":
                return block["input"].get("pages", [])
        return None

    def extract_concepts(self, system: str, prompt: str) -> Optional[list[dict]]:
        return self._retry_call(lambda: self._extract_concepts(system, prompt))

    def _extract_concepts(self, system: str, prompt: str) -> Optional[list[dict]]:
        tool_def = {
            "name": "extract_concepts",
            "description": "Extract key concepts from a source document",
            "input_schema": {
                "type": "object",
                "properties": {
                    "concepts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "concept": {"type": "string"},
                                "summary": {"type": "string"},
                                "is_new": {"type": "boolean"},
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "confidence": {"type": "number"},
                                "provenance_state": {
                                    "type": "string",
                                    "enum": ["extracted", "merged", "inferred", "ambiguous"],
                                },
                                "contradicted_by": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "slug": {"type": "string"},
                                            "reason": {"type": "string"},
                                        },
                                        "required": ["slug", "reason"],
                                    },
                                },
                            },
                            "required": ["concept", "summary", "is_new"],
                        },
                    }
                },
                "required": ["concepts"],
            },
        }
        body = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [tool_def],
            "tool_choice": {"type": "tool", "name": "extract_concepts"},
        }
        data = self._request(body)
        for block in data.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "extract_concepts":
                return block["input"].get("concepts", [])
        return None


class OpenAIProvider(BaseProvider):
    def __init__(self):
        self.base_url = LLMWIKI_BASE_URL or "http://localhost:8000/v1"
        self.model = LLMWIKI_MODEL
        self.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.embed_model = LLMWIKI_EMBEDDING_MODEL or self.model
        self.embed_url = (LLMWIKI_EMBEDDING_BASE_URL or self.base_url).rstrip("/") + "/embeddings"

    def _retry_call(self, fn):
        last_ex = None
        for attempt in range(RETRY_COUNT):
            try:
                return fn()
            except LLMError as e:
                last_ex = e
                resp_code = getattr(e, "resp_code", 0)
                if 400 <= resp_code < 500 and resp_code != 429:
                    raise
                if attempt < RETRY_COUNT - 1:
                    delay = random.uniform(RETRY_BASE_MS / 2000, RETRY_BASE_MS / 1000) * (RETRY_MULTIPLIER ** attempt)
                    time.sleep(delay)
        raise last_ex

    def _request(self, body: dict) -> dict:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=300,
        )
        if resp.status_code != 200:
            e = LLMError(f"OpenAI API error {resp.status_code}: {resp.text}")
            e.resp_code = resp.status_code
            raise e
        return resp.json()

    def complete(self, system: str, prompt: str) -> str:
        return self._retry_call(lambda: self._complete(system, prompt))

    def _complete(self, system: str, prompt: str) -> str:
        body = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        data = self._request(body)
        content = data["choices"][0]["message"].get("content", "")
        return _strip_think(content)

    def embed(self, text: str, input_type: str = "document") -> list[float]:
        resp = requests.post(
            self.embed_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.embed_model, "input": text},
            timeout=120,
        )
        if resp.status_code != 200:
            raise LLMError(f"Embedding API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["data"][0]["embedding"]

    def select_pages(self, system: str, prompt: str, max_items: int = 5) -> Optional[list[str]]:
        return self._retry_call(lambda: self._select_pages(system, prompt, max_items))

    def _select_pages(self, system: str, prompt: str, max_items: int) -> Optional[list[str]]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "select_pages",
                    "description": "Select relevant wiki pages for a question",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pages": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": max_items,
                                "description": "Slugs of relevant pages",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Explanation of selection",
                            },
                        },
                        "required": ["pages", "reasoning"],
                    },
                },
            }
        ]
        body = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "select_pages"}},
        }
        data = self._request(body)
        msg = data["choices"][0]["message"]
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc["function"]["name"] == "select_pages":
                    args = json.loads(tc["function"]["arguments"])
                    return args.get("pages", [])
        return None

    def extract_concepts(self, system: str, prompt: str) -> Optional[list[dict]]:
        return self._retry_call(lambda: self._extract_concepts(system, prompt))

    def _extract_concepts(self, system: str, prompt: str) -> Optional[list[dict]]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_concepts",
                    "description": "Extract key concepts from a source document",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "concepts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "concept": {"type": "string"},
                                        "summary": {"type": "string"},
                                        "is_new": {"type": "boolean"},
                                        "tags": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "confidence": {"type": "number"},
                                        "provenance_state": {
                                            "type": "string",
                                            "enum": ["extracted", "merged", "inferred", "ambiguous"],
                                        },
                                        "contradicted_by": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "slug": {"type": "string"},
                                                    "reason": {"type": "string"},
                                                },
                                                "required": ["slug", "reason"],
                                            },
                                        },
                                    },
                                    "required": ["concept", "summary", "is_new"],
                                },
                            }
                        },
                        "required": ["concepts"],
                    },
                },
            }
        ]
        body = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "extract_concepts"}},
        }
        data = self._request(body)
        msg = data["choices"][0]["message"]
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc["function"]["name"] == "extract_concepts":
                    args = json.loads(tc["function"]["arguments"])
                    return args.get("concepts", [])
        return None


_PROVIDER_INSTANCE: Optional[BaseProvider] = None


def get_provider() -> BaseProvider:
    global _PROVIDER_INSTANCE
    if _PROVIDER_INSTANCE is not None:
        return _PROVIDER_INSTANCE
    if LLMWIKI_PROVIDER == "openai":
        _PROVIDER_INSTANCE = OpenAIProvider()
    else:
        _PROVIDER_INSTANCE = AnthropicProvider()
    return _PROVIDER_INSTANCE


def set_provider(provider: BaseProvider) -> None:
    global _PROVIDER_INSTANCE
    _PROVIDER_INSTANCE = provider
