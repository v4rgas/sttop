"""Cloud transcription against any OpenAI-compatible /audio/transcriptions API.

Works with Groq (default), OpenAI, and self-hosted whisper servers that copy
the same route. Note that OpenRouter is *not* usable here: it proxies chat
completions only and exposes no audio-transcription endpoint.
"""

from __future__ import annotations

import os

import httpx

from ..config import CloudConfig, SttConfig
from .base import Transcript, pcm_to_wav


class CloudTranscriber:
    def __init__(self, config: CloudConfig, stt: SttConfig) -> None:
        self.config = config
        self.stt = stt

        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"${config.api_key_env} is not set - export it or switch "
                "stt.backend back to 'local'"
            )

        host = httpx.URL(config.base_url).host
        self.describe = f"cloud {config.model} @ {host}"
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=config.timeout_s,
        )

    def transcribe(self, pcm: bytes) -> Transcript:
        data = {"model": self.config.model, "response_format": "json"}
        if self.stt.language:
            data["language"] = self.stt.language

        response = self._client.post(
            "/audio/transcriptions",
            files={"file": ("segment.wav", pcm_to_wav(pcm), "audio/wav")},
            data=data,
        )
        response.raise_for_status()
        payload = response.json()
        return Transcript(
            text=(payload.get("text") or "").strip(),
            language=payload.get("language") or self.stt.language,
        )

    def close(self) -> None:
        self._client.close()
