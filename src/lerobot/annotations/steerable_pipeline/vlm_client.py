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
import os
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
    if config.backend == "openai":
        return _make_openai_client(config)
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
        # ``guided_decoding`` would speed up parsing but its API differs across
        # vllm releases (dict vs GuidedDecodingParams). The _GenericTextClient
        # wrapper already has a one-retry JSON-recovery path, so we skip it.
        params = SamplingParams(max_tokens=max_tok, temperature=temp)
        # ``llm.chat`` handles chat-template application + multimodal input
        # extraction (image/video blocks) internally, which ``llm.generate``
        # does not.
        outputs = llm.chat([list(m) for m in batch], params)
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


def _make_openai_client(config: VlmConfig) -> VlmClient:
    """Backend that talks to any OpenAI-compatible server.

    Compatible with ``vllm serve``, ``transformers serve``,
    ``ktransformers serve``, and hosted endpoints. By default the server
    is expected to be already running. Set ``auto_serve=True`` to have
    this client spawn one (default: ``transformers serve``), wait until
    it's ready, and tear it down on process exit.

    Image blocks ``{"type":"image", "image":<PIL.Image>}`` are
    auto-converted to ``image_url`` data-URLs. Video blocks
    ``{"type":"video", "video":[<PIL>...]}`` are forwarded as
    multi-frame ``video_url`` items where supported.
    """
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "openai package is required for backend='openai'. "
            "Install with `pip install openai`."
        ) from exc

    api_base = config.api_base
    api_key = config.api_key
    auto_serve = config.auto_serve

    if config.use_hf_inference_providers:
        api_base = "https://router.huggingface.co/v1"
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or ""
        if not token:
            raise RuntimeError(
                "use_hf_inference_providers=True requires HF_TOKEN (or "
                "HUGGINGFACE_API_KEY) in the environment."
            )
        api_key = token
        auto_serve = False
        print(
            f"[lerobot-annotate] HF Inference Providers: routing model={config.model_id} "
            f"via {api_base}",
            flush=True,
        )
    else:
        print(
            f"[lerobot-annotate] backend=openai model={config.model_id} "
            f"api_base={api_base} auto_serve={auto_serve}",
            flush=True,
        )
        if auto_serve:
            if _server_is_up(api_base):
                print(f"[lerobot-annotate] reusing server already up at {api_base}", flush=True)
            else:
                print("[lerobot-annotate] no server reachable; spawning one", flush=True)
                api_base = _spawn_inference_server(config)
                print(f"[lerobot-annotate] server ready at {api_base}", flush=True)

    client = OpenAI(base_url=api_base, api_key=api_key)

    # ``mm_processor_kwargs`` is a vllm-specific extra; transformers serve
    # rejects it with HTTP 422. Send it only when explicitly opted in via
    # an env var (e.g. ``LEROBOT_OPENAI_SEND_MM_KWARGS=1`` for vllm).
    send_mm_kwargs = os.environ.get(
        "LEROBOT_OPENAI_SEND_MM_KWARGS", ""
    ).lower() in {"1", "true", "yes"}

    def _gen(
        batch: Sequence[Sequence[dict[str, Any]]], max_tok: int, temp: float
    ) -> list[str]:
        outs: list[str] = []
        for messages in batch:
            api_messages, mm_kwargs = _to_openai_messages(messages)
            kwargs: dict[str, Any] = {
                "model": config.model_id,
                "messages": api_messages,
                "max_tokens": max_tok,
                "temperature": temp,
            }
            if send_mm_kwargs and mm_kwargs:
                kwargs["extra_body"] = {
                    "mm_processor_kwargs": {**mm_kwargs, "do_sample_frames": True}
                }
            response = client.chat.completions.create(**kwargs)
            outs.append(response.choices[0].message.content or "")
        return outs

    return _GenericTextClient(_gen, config)


def _server_is_up(api_base: str) -> bool:
    """Return True if ``api_base/models`` answers 200 within 2 seconds."""
    import urllib.request  # noqa: PLC0415

    url = api_base.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def _spawn_inference_server(config: VlmConfig) -> str:
    """Spawn ``transformers serve`` (or ``serve_command``), wait until it
    accepts ``/v1/models``, and register a shutdown hook.

    Streams the server's stdout/stderr to the parent terminal in
    real-time on a background thread so users can see model-load
    progress and errors as they happen.

    Returns the full ``api_base`` URL the OpenAI client should use.
    """
    import atexit  # noqa: PLC0415
    import shlex  # noqa: PLC0415
    import signal  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import threading  # noqa: PLC0415
    import time  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    cmd = config.serve_command
    if not cmd:
        cmd = (
            f"transformers serve {shlex.quote(config.model_id)} "
            f"--port {config.serve_port} --continuous-batching"
        )
    api_base = f"http://localhost:{config.serve_port}/v1"
    print(f"[server] launching: {cmd}", flush=True)
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Watch the server output for the uvicorn readiness banner. This is
    # more reliable than polling /v1/models because transformers serve
    # rescans its cache on every model-list request, which can exceed
    # the urllib timeout and trigger an infinite probe loop.
    ready_event = threading.Event()
    ready_markers = ("Uvicorn running", "Application startup complete")

    def _stream_output() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(f"[server] {line}")
            sys.stdout.flush()
            if any(marker in line for marker in ready_markers):
                ready_event.set()

    threading.Thread(target=_stream_output, daemon=True).start()

    def _shutdown() -> None:
        if proc.poll() is None:
            print(f"[server] stopping pid={proc.pid}", flush=True)
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    atexit.register(_shutdown)

    deadline = time.monotonic() + config.serve_ready_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"[server] inference server exited unexpectedly with rc={proc.returncode}. "
                f"See [server] log lines above for the cause."
            )
        if ready_event.wait(timeout=2):
            return api_base
    proc.terminate()
    raise RuntimeError(
        f"[server] did not become ready within {config.serve_ready_timeout_s}s"
    )


def _to_openai_messages(
    messages: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert internal messages to OpenAI chat format.

    Returns ``(api_messages, mm_kwargs)``. Multimodal-processor kwargs
    (``fps`` from ``video_url`` blocks) are extracted out so the caller
    can pass them via ``extra_body.mm_processor_kwargs`` rather than
    inside the content blocks (which transformers serve rejects).

    File-URL video blocks are inlined as base64 data URLs.
    """
    out_messages: list[dict[str, Any]] = []
    mm_kwargs: dict[str, Any] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out_messages.append({"role": message["role"], "content": content})
            continue
        out_blocks: list[dict[str, Any]] = []
        for block in content:
            block_type = block.get("type") if isinstance(block, dict) else None
            if block_type == "text":
                out_blocks.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "image":
                out_blocks.append(
                    {"type": "image_url", "image_url": {"url": _pil_to_data_url(block["image"])}}
                )
            elif block_type == "video":
                frames = block.get("video", [])
                for img in frames:
                    out_blocks.append(
                        {"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}}
                    )
            elif block_type == "video_url":
                video_url = dict(block["video_url"])
                url = video_url.get("url", "")
                if url.startswith("file://"):
                    video_url["url"] = _file_to_data_url(url[len("file://") :])
                out_blocks.append({"type": "video_url", "video_url": video_url})
                fps = block.get("fps")
                if fps is not None:
                    mm_kwargs["fps"] = fps
            else:
                out_blocks.append(block)
        out_messages.append({"role": message["role"], "content": out_blocks})
    return out_messages, mm_kwargs


def _file_to_data_url(path: str) -> str:
    """Read a local video file and return a base64 ``data:video/mp4`` URL."""
    import base64  # noqa: PLC0415

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


def _pil_to_data_url(image: Any) -> str:
    """Encode a PIL.Image as a base64 data URL."""
    import base64  # noqa: PLC0415
    import io  # noqa: PLC0415

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _messages_to_prompt(messages: Sequence[dict[str, Any]]) -> Any:
    """Pass-through hook used by the vllm backend.

    vllm exposes its own multimodal entry points that vary by version; for the
    base flow we simply forward the raw message list and let the caller's
    custom backend handle templating. Real deployments override this.
    """
    return list(messages)
