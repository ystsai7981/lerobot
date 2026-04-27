#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared Qwen-VL client.

The pipeline uses a single shared VLM across modules. vLLM is preferred when
available (high throughput, JSON-guided decoding); transformers is the
fallback. A ``stub`` backend is used for unit tests so fixtures never call
into a real model.

The client speaks one method, :meth:`VlmClient.generate_json`, which:

- accepts a list of OpenAI/HF-style multimodal messages,
- requests JSON output (``json_mode=True`` enables guided decoding when the
  backend supports it),
- batches requests transparently,
- and reprompts once on a JSON parse failure with an inline correction
  message before raising.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .config import VlmConfig


class VlmClient(Protocol):
    """Protocol every backend must implement."""

    def generate_json(
        self,
        messages_batch: Sequence[Sequence[dict[str, Any]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Any]:
        """Generate one JSON-decoded response per messages list."""


@dataclass
class StubVlmClient:
    """Deterministic stub used in unit tests.

    A test passes a callable that maps the *last user message text* (or, if
    that is empty, the full message list) to a JSON-serializable response.
    """

    responder: Callable[[Sequence[dict[str, Any]]], Any]

    def generate_json(
        self,
        messages_batch: Sequence[Sequence[dict[str, Any]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Any]:
        return [self.responder(list(messages)) for messages in messages_batch]


def _strip_to_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        # tolerate ```json ... ``` fences from chat-tuned backbones
        first = text.find("\n")
        last = text.rfind("```")
        if first != -1 and last != -1 and last > first:
            text = text[first + 1 : last].strip()
    return json.loads(text)


@dataclass
class _GenericTextClient:
    """Wraps any text-generation callable in JSON-mode + one-retry semantics."""

    generate_text: Callable[[Sequence[Sequence[dict[str, Any]]], int, float], list[str]]
    config: VlmConfig

    def generate_json(
        self,
        messages_batch: Sequence[Sequence[dict[str, Any]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Any]:
        max_tok = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        temp = temperature if temperature is not None else self.config.temperature
        raw = self.generate_text(messages_batch, max_tok, temp)
        out: list[Any] = []
        for messages, text in zip(messages_batch, raw, strict=True):
            try:
                out.append(_strip_to_json(text))
                continue
            except (ValueError, json.JSONDecodeError):
                pass
            retry = list(messages) + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON. "
                        "Reply with strictly valid JSON, no prose, no fences."
                    ),
                },
            ]
            retry_text = self.generate_text([retry], max_tok, temp)[0]
            out.append(_strip_to_json(retry_text))
        return out


def make_vlm_client(config: VlmConfig) -> VlmClient:
    """Build the shared VLM client per the configured backend.

    For ``stub``, callers should construct :class:`StubVlmClient` directly with
    a responder callable. ``stub`` here is rejected to make accidental misuse
    obvious.
    """
    if config.backend == "stub":
        raise ValueError(
            "Use StubVlmClient(...) directly for the stub backend; make_vlm_client builds real clients."
        )
    if config.backend == "vllm":
        return _make_vllm_client(config)
    if config.backend == "transformers":
        return _make_transformers_client(config)
    raise ValueError(f"Unknown VLM backend: {config.backend!r}")


def _make_vllm_client(config: VlmConfig) -> VlmClient:
    try:
        from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "vllm is required for backend='vllm'. Install with `pip install lerobot[annotations]`."
        ) from exc
    llm = LLM(model=config.model_id, tensor_parallel_size=config.tensor_parallel_size)

    def _gen(batch: Sequence[Sequence[dict[str, Any]]], max_tok: int, temp: float) -> list[str]:
        params = SamplingParams(
            max_tokens=max_tok,
            temperature=temp,
            guided_decoding={"json": {}} if config.json_mode else None,
        )
        prompts = [_messages_to_prompt(m) for m in batch]
        outputs = llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

    return _GenericTextClient(_gen, config)


def _make_transformers_client(config: VlmConfig) -> VlmClient:
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("transformers + torch are required for backend='transformers'.") from exc
    processor = AutoProcessor.from_pretrained(config.model_id)
    model = AutoModelForVision2Seq.from_pretrained(config.model_id, torch_dtype="auto")
    model.eval()

    def _gen(batch: Sequence[Sequence[dict[str, Any]]], max_tok: int, temp: float) -> list[str]:
        outs: list[str] = []
        for messages in batch:
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=[text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_tok,
                    temperature=temp,
                    do_sample=temp > 0.0,
                )
            decoded = processor.batch_decode(
                gen[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )[0]
            outs.append(decoded)
        return outs

    return _GenericTextClient(_gen, config)


def _messages_to_prompt(messages: Sequence[dict[str, Any]]) -> Any:
    """Pass-through hook used by the vllm backend.

    vllm exposes its own multimodal entry points that vary by version; for the
    base flow we simply forward the raw message list and let the caller's
    custom backend handle templating. Real deployments override this.
    """
    return list(messages)
