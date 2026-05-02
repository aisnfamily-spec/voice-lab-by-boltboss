"""
Voice Lab — minimal multi-engine TTS CLI.

Two engines, one CLI:

  --engine elevenlabs   (default)  — paid API, premium quality
                                     ENV: ELEVENLABS_API_KEY, VOICE_ID
  --engine piper                   — open-source, local, runs offline,
                                     no API key, no internet calls

Both engines:
- sentence-aware chunking
- async parallel workers (Piper falls back to sequential — single
  ONNX model can't be safely parallelised across threads)
- pydub-stitched MP3 output

Deliberately NOT INCLUDED in the public version:
- Firebase free-tier login flow
- Proxy rotation / IP burnt handling
- HCaptcha / anti-abuse bypass
- Multi-account quarantine logic

If ElevenLabs rate-limits you, drop --workers or upgrade your plan.
If Piper is too slow, use a smaller model (`x_low` < `low` < `medium` < `high`).
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import io
import os
import re
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import aiohttp
from dotenv import load_dotenv
from pydub import AudioSegment


ELEVENLABS_API_BASE = "https://api.elevenlabs.io"
DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_WORKERS = 4
DEFAULT_REQUEST_TIMEOUT = 90.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_S = 5.0


# ---------------------------------------------------------------------------
# Sentence-aware chunker (shared by both engines).
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r'(?<=[a-zа-яё0-9][.!?])\s+(?=[A-ZА-ЯЁ«"(\[])')
_SENTENCE_FALLBACK = re.compile(r'(?<=[.!?])\s+')


def split_text(text: str, chunk_size: int) -> list[str]:
    """Break text into <= chunk_size pieces, splitting on sentence boundaries."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sentences = _SENTENCE_SPLIT.split(text)
    if len(sentences) <= 1:
        sentences = _SENTENCE_FALLBACK.split(text)
    if not sentences:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sent in sentences:
        sl = len(sent) + 1
        if size + sl > chunk_size and current:
            chunks.append(" ".join(current))
            current = []
            size = 0
        current.append(sent)
        size += sl
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------

class TTSError(Exception):
    pass


class QuotaExhausted(TTSError):
    pass


class ContentBlocked(TTSError):
    pass


# ---------------------------------------------------------------------------
# ElevenLabs engine — paid xi-api-key, no proxy / firebase / captcha.
# ---------------------------------------------------------------------------

async def synth_chunk_elevenlabs(
    session: aiohttp.ClientSession,
    *,
    api_key: str,
    text: str,
    voice_id: str,
    model: str,
    stability: float,
    similarity_boost: float,
    style: float,
    speed: float,
    timeout: float,
    max_retries: int,
    backoff_s: float,
) -> bytes:
    """Single chunk → MP3 bytes via ElevenLabs streaming endpoint."""
    url = f"{ELEVENLABS_API_BASE}/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": model,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style_exaggeration": style,
        },
    }
    if speed != 1.0:
        payload["generation_config"] = {"speed": speed}

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
                body = (await resp.text())[:300]
                if resp.status == 401:
                    raise TTSError(f"401 unauthorized — check xi-api-key: {body}")
                if resp.status in (402, 403):
                    raise QuotaExhausted(f"{resp.status} — quota / forbidden: {body}")
                if resp.status == 422:
                    raise ContentBlocked(f"422 content rejected: {body}")
                if resp.status in (429, 500, 502, 503, 504):
                    last_err = TTSError(f"{resp.status} {body}")
                    if attempt < max_retries:
                        wait = backoff_s * attempt
                        print(
                            f"  [retry {attempt}/{max_retries}] HTTP {resp.status} → "
                            f"sleep {wait:.0f}s",
                            file=sys.stderr,
                        )
                        await asyncio.sleep(wait)
                        continue
                raise TTSError(f"unexpected HTTP {resp.status}: {body}")
        except (TTSError, QuotaExhausted, ContentBlocked):
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_err = e
            if attempt < max_retries:
                wait = backoff_s * attempt
                print(
                    f"  [retry {attempt}/{max_retries}] {type(e).__name__}: {e} → "
                    f"sleep {wait:.0f}s",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
                continue
            break
    raise TTSError(f"giving up after {max_retries} attempts: {last_err}")


async def synthesize_elevenlabs(
    *,
    chunks: list[str],
    output_path: Path,
    api_key: str,
    voice_id: str,
    model: str,
    stability: float,
    similarity_boost: float,
    style: float,
    speed: float,
    workers: int,
    timeout: float,
    max_retries: int,
    backoff_s: float,
) -> None:
    print(
        f"engine : elevenlabs\n"
        f"voice  : {voice_id}\n"
        f"model  : {model}\n"
        f"workers: {min(workers, len(chunks))}",
    )

    results: dict[int, bytes] = {}
    queue: asyncio.Queue = asyncio.Queue()
    for i, c in enumerate(chunks):
        await queue.put((i, c))
    num_workers = min(workers, len(chunks))
    for _ in range(num_workers):
        await queue.put(None)

    async with aiohttp.ClientSession() as session:
        async def worker(wid: int) -> None:
            while True:
                item = await queue.get()
                if item is None:
                    return
                idx, chunk_text = item
                try:
                    audio = await synth_chunk_elevenlabs(
                        session,
                        api_key=api_key,
                        text=chunk_text,
                        voice_id=voice_id,
                        model=model,
                        stability=stability,
                        similarity_boost=similarity_boost,
                        style=style,
                        speed=speed,
                        timeout=timeout,
                        max_retries=max_retries,
                        backoff_s=backoff_s,
                    )
                    results[idx] = audio
                    print(f"  W{wid}: chunk {idx + 1}/{len(chunks)} ✓")
                except (QuotaExhausted, ContentBlocked):
                    raise
                except TTSError as e:
                    print(f"  W{wid}: chunk {idx + 1} FAILED: {e}", file=sys.stderr)
                    results[idx] = b""

        tasks = [asyncio.create_task(worker(i)) for i in range(num_workers)]
        await asyncio.gather(*tasks)

    missing = [i + 1 for i in range(len(chunks)) if i not in results or not results[i]]
    if missing:
        raise TTSError(f"failed chunks: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(chunks) == 1:
        output_path.write_bytes(results[0])
    else:
        combined = AudioSegment.empty()
        for i in range(len(chunks)):
            combined += AudioSegment.from_mp3(io.BytesIO(results[i]))
        combined.export(str(output_path), format="mp3", bitrate="192k")


# ---------------------------------------------------------------------------
# Piper engine — open-source, local, no API.
# ---------------------------------------------------------------------------

def _load_piper_voice(model_path: Path):
    """Lazy-import + cache.  PiperVoice instances are NOT thread-safe; we
    serialise synth calls through a single-thread executor."""
    try:
        from piper.voice import PiperVoice  # type: ignore
    except ImportError as e:
        raise TTSError(
            "Piper not installed.  pip install piper-tts  (and download a voice)"
        ) from e
    if not model_path.exists():
        raise TTSError(f"Piper model not found: {model_path}")
    config_path = model_path.with_suffix(model_path.suffix + ".json")
    if not config_path.exists():
        raise TTSError(
            f"Piper config not found next to model: {config_path}\n"
            "Both .onnx and .onnx.json must sit in the same folder."
        )
    return PiperVoice.load(str(model_path), config_path=str(config_path))


def _piper_chunk_to_wav(voice, text: str, length_scale: float) -> bytes:
    """Synthesise one chunk to in-memory WAV (sync, runs in executor)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        voice.synthesize(
            text,
            wf,
            length_scale=length_scale,
        )
    return buffer.getvalue()


async def synthesize_piper(
    *,
    chunks: list[str],
    output_path: Path,
    model_path: Path,
    length_scale: float,
) -> None:
    print(
        f"engine : piper\n"
        f"model  : {model_path.name}\n"
        f"workers: 1 (Piper voice not thread-safe)",
    )
    voice = _load_piper_voice(model_path)
    loop = asyncio.get_running_loop()

    results: list[bytes] = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for i, chunk_text in enumerate(chunks):
            wav_bytes = await loop.run_in_executor(
                pool, functools.partial(
                    _piper_chunk_to_wav, voice, chunk_text, length_scale,
                ),
            )
            results.append(wav_bytes)
            print(f"  chunk {i + 1}/{len(chunks)} ✓")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = AudioSegment.empty()
    for wav_bytes in results:
        combined += AudioSegment.from_wav(io.BytesIO(wav_bytes))
    combined.export(str(output_path), format="mp3", bitrate="192k")


# ---------------------------------------------------------------------------
# Top-level dispatch.
# ---------------------------------------------------------------------------

async def synthesize(
    *,
    text: str,
    output_path: Path,
    engine: str,
    chunk_size: int,
    workers: int,
    timeout: float,
    max_retries: int,
    backoff_s: float,
    # ElevenLabs
    api_key: Optional[str] = None,
    voice_id: Optional[str] = None,
    el_model: str = DEFAULT_ELEVENLABS_MODEL,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.3,
    speed: float = 1.0,
    # Piper
    piper_model: Optional[Path] = None,
    piper_length_scale: float = 1.0,
) -> None:
    chunks = split_text(text, chunk_size)
    if not chunks:
        raise ValueError("input text is empty after stripping")

    print(
        f"text   : {len(text)} chars → {len(chunks)} chunks "
        f"(<= {chunk_size} chars each)",
    )

    if engine == "elevenlabs":
        if not api_key:
            raise TTSError("ELEVENLABS_API_KEY missing — see .env.example")
        if not voice_id:
            raise TTSError("--voice or VOICE_ID required for elevenlabs engine")
        await synthesize_elevenlabs(
            chunks=chunks,
            output_path=output_path,
            api_key=api_key,
            voice_id=voice_id,
            model=el_model,
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            speed=speed,
            workers=workers,
            timeout=timeout,
            max_retries=max_retries,
            backoff_s=backoff_s,
        )
    elif engine == "piper":
        if not piper_model:
            raise TTSError(
                "--piper-model required for piper engine "
                "(e.g. --piper-model voices/en_US-amy-medium.onnx)"
            )
        await synthesize_piper(
            chunks=chunks,
            output_path=output_path,
            model_path=piper_model,
            length_scale=piper_length_scale,
        )
    else:
        raise TTSError(f"unknown engine: {engine!r} — use 'elevenlabs' or 'piper'")

    size_kb = output_path.stat().st_size / 1024
    print(f"\n✓ {output_path}  ({size_kb:,.1f} KB)")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Voice Lab — TTS CLI (ElevenLabs API or local Piper).",
    )
    ap.add_argument("input", help="path to .txt file with the script")
    ap.add_argument(
        "-o", "--output", default="output.mp3",
        help="output MP3 path (default: output.mp3)",
    )
    ap.add_argument(
        "--engine", choices=["elevenlabs", "piper"],
        default=os.getenv("ENGINE", "elevenlabs"),
        help="which TTS backend to use (default: elevenlabs, env: ENGINE)",
    )
    # Shared
    ap.add_argument(
        "--chunk", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"chunk size in chars (default: {DEFAULT_CHUNK_SIZE})",
    )
    ap.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"parallel workers, ElevenLabs only (default: {DEFAULT_WORKERS})",
    )
    # ElevenLabs
    ap.add_argument(
        "--voice", default=os.getenv("VOICE_ID"),
        help="ElevenLabs voice_id (env: VOICE_ID)",
    )
    ap.add_argument(
        "--model", default=os.getenv("MODEL", DEFAULT_ELEVENLABS_MODEL),
        help=f"ElevenLabs model_id (default: {DEFAULT_ELEVENLABS_MODEL})",
    )
    ap.add_argument("--stability", type=float, default=0.5)
    ap.add_argument("--similarity", type=float, default=0.75)
    ap.add_argument("--style", type=float, default=0.3)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument(
        "--timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT,
        help=f"per-request timeout in seconds (default: {DEFAULT_REQUEST_TIMEOUT})",
    )
    ap.add_argument(
        "--retries", type=int, default=DEFAULT_MAX_RETRIES,
        help=f"max retries per chunk (default: {DEFAULT_MAX_RETRIES})",
    )
    ap.add_argument(
        "--backoff", type=float, default=DEFAULT_BACKOFF_S,
        help=f"linear backoff base seconds (default: {DEFAULT_BACKOFF_S})",
    )
    # Piper
    ap.add_argument(
        "--piper-model", type=Path, default=None,
        help="path to Piper .onnx model file (its .onnx.json must sit next to it)",
    )
    ap.add_argument(
        "--piper-length-scale", type=float, default=1.0,
        help="speak rate, >1.0 slower / <1.0 faster (default: 1.0)",
    )
    return ap.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        return 2
    text = in_path.read_text(encoding="utf-8")

    api_key = os.getenv("ELEVENLABS_API_KEY")

    try:
        asyncio.run(synthesize(
            text=text,
            output_path=Path(args.output),
            engine=args.engine,
            chunk_size=args.chunk,
            workers=args.workers,
            timeout=args.timeout,
            max_retries=args.retries,
            backoff_s=args.backoff,
            # ElevenLabs
            api_key=api_key,
            voice_id=args.voice,
            el_model=args.model,
            stability=args.stability,
            similarity_boost=args.similarity,
            style=args.style,
            speed=args.speed,
            # Piper
            piper_model=args.piper_model,
            piper_length_scale=args.piper_length_scale,
        ))
    except (TTSError, QuotaExhausted, ContentBlocked) as e:
        print(f"\nFAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
