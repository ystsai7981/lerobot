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
    # Workaround for cuDNN 9.x + torch 2.8 conv3d regression that surfaces
    # as CUDNN_STATUS_NOT_INITIALIZED in Qwen-VL vision-tower patch
    # embedders. Setting LEROBOT_DISABLE_CUDNN=1 forces native PyTorch
    # convolution kernels — slower but functional.
    import os as _os  # noqa: PLC0415

    if _os.environ.get("LEROBOT_DISABLE_CUDNN", "").lower() in {"1", "true", "yes"}:
        import torch as _torch  # noqa: PLC0415

        _torch.backends.cudnn.enabled = False
    llm_kwargs: dict[str, Any] = {
        "model": config.model_id,
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.max_model_len is not None:
        llm_kwargs["max_model_len"] = config.max_model_len
    llm = LLM(**llm_kwargs)

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
        import transformers  # type: ignore[import-not-found]
        from transformers import AutoProcessor  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("transformers + torch are required for backend='transformers'.") from exc
    auto_cls = (
        getattr(transformers, "AutoModelForImageTextToText", None)
        or getattr(transformers, "AutoModelForVision2Seq", None)
    )
    if auto_cls is None:
        raise ImportError(
            "Neither AutoModelForImageTextToText nor AutoModelForVision2Seq is available in this "
            "transformers version. Install transformers>=4.45 (which has AutoModelForImageTextToText) "
            "for VL models."
        )
    processor = AutoProcessor.from_pretrained(
        config.model_id, trust_remote_code=config.trust_remote_code
    )
    import os as _os  # noqa: PLC0415

    use_accelerate = _os.environ.get("LEROBOT_TRANSFORMERS_DEVICE_MAP", "manual") != "manual"
    # ``device_map='auto'`` triggers a known std::bad_alloc on the Qwen3-VL
    # post-load dispatch path (the alloc fails in accelerate's hook setup
    # even with TBs of host RAM). Default to manual: load on CPU with
    # ``low_cpu_mem_usage=True``, then ``.to("cuda")``. Set
    # ``LEROBOT_TRANSFORMERS_DEVICE_MAP=auto`` to opt back into the old path.
    if use_accelerate:
        model = auto_cls.from_pretrained(
            config.model_id,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=config.trust_remote_code,
        )
    else:
        import torch as _torch  # noqa: PLC0415

        model = auto_cls.from_pretrained(
            config.model_id,
            torch_dtype=_torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=config.trust_remote_code,
        )
        model = model.to("cuda")
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
