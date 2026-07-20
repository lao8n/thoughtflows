"""LLM layer — driven through the Claude Agent SDK.

This runs the `claude` binary under your existing Claude Code login, so there is
NO separate ANTHROPIC_API_KEY to manage (architecture §9: Claude Agent SDK).
The loop lives in our code (explore.py); this is one structured-output step
inside it. Model-internal knowledge only for now — no tools / grounding yet.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Type, TypeVar

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from pydantic import BaseModel

# None → use whatever model Claude Code is currently set to (Opus by default).
DEFAULT_MODEL = os.getenv("FORECASTER_MODEL") or None

# Model tiering (architecture §14): a FAST model for high-volume mining (explore),
# a DEEP model for judgment-heavy stages (decision graph, gate tracker, synthesis).
FAST_MODEL = os.getenv("FORECASTER_FAST_MODEL", "claude-sonnet-4-6")
DEEP_MODEL = os.getenv("FORECASTER_DEEP_MODEL") or None  # None → CLI default (Opus)

T = TypeVar("T", bound=BaseModel)


def _json_schema(model: Type[BaseModel]) -> dict[str, Any]:
    """Pydantic → a structured-outputs-compatible JSON schema.

    The Messages API (and the Agent SDK's output_format) requires
    additionalProperties:false on every object; Pydantic omits it.
    """
    schema = model.model_json_schema()
    _force_no_additional(schema)
    return schema


def _force_no_additional(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and "additionalProperties" not in node:
            node["additionalProperties"] = False
        for v in node.values():
            _force_no_additional(v)
    elif isinstance(node, list):
        for v in node:
            _force_no_additional(v)


def _extract_json(text: str) -> Any:
    """Best-effort: pull the outermost JSON object/array from a text blob."""
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no parseable JSON found in model output")


class LLM:
    def __init__(self, model: str | None = DEFAULT_MODEL) -> None:
        self.model = model
        self.last_cost_usd: float | None = None

    def parse(self, *, system: str, user: str, schema: Type[T], retries: int = 2,
              tools: list[str] | None = None, max_turns: int = 8,
              call_timeout: int = 600, max_buffer_size: int | None = None) -> T:
        """One structured-output call. Returns a validated `schema` instance.

        Retries on transient CLI/stream errors (the loop makes many calls).
        `tools`: server tools to allow (e.g. ["WebSearch"]) — for grounded stages.
        If a tool is unavailable in the environment, the model degrades gracefully
        to internal knowledge (the prompt must ask it to flag grounding honestly).
        """
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return asyncio.run(self._parse(system=system, user=user, schema=schema,
                                               tools=tools, max_turns=max_turns,
                                               call_timeout=call_timeout,
                                               max_buffer_size=max_buffer_size))
            except Exception as e:  # transient CLI/rate-limit/stream hiccups
                last = e
                if attempt < retries:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM.parse failed after {retries + 1} attempts: {last}")

    async def _parse(self, *, system: str, user: str, schema: Type[T],
                     tools: list[str] | None = None, max_turns: int = 8,
                     call_timeout: int = 600, max_buffer_size: int | None = None) -> T:
        """call_timeout: max seconds for a single API call before we cancel and retry.
        Default 10 min — long enough for deep Opus thinking + a few web searches.
        max_buffer_size: bytes for the SDK stream reader; raise it for tool-grounded
        stages where a single web-search result can exceed the 1 MB default."""
        opt_kwargs = dict(
            system_prompt=system,
            model=self.model,
            allowed_tools=tools or [],  # [] → pure reasoning; ["WebSearch"] → grounded
            max_turns=max_turns,
            thinking={"type": "adaptive"},
            output_format={"type": "json_schema", "schema": _json_schema(schema)},
        )
        if max_buffer_size is not None:
            opt_kwargs["max_buffer_size"] = max_buffer_size
        options = ClaudeAgentOptions(**opt_kwargs)

        async def _run() -> T:
            structured: Any = None
            text_parts: list[str] = []
            async for msg in query(prompt=user, options=options):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                elif isinstance(msg, ResultMessage):
                    structured = msg.structured_output
                    self.last_cost_usd = msg.total_cost_usd

            if structured is not None:
                return schema.model_validate(structured)
            return schema.model_validate(_extract_json("".join(text_parts)))

        # Wrap in a timeout so hung calls fail fast and trigger the retry logic.
        return await asyncio.wait_for(_run(), timeout=call_timeout)
