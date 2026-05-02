"""
ElevenLabs TTS — minimal standalone CLI.

A clean, public-friendly TTS script:
- One paid ElevenLabs API key (xi-api-key auth)
- Sentence-aware chunking for long scripts
- N parallel workers
- MP3 stitched with pydub

NOT INCLUDED (left out on purpose for the public version):
- Firebase free-tier login flow
- Proxy rotation / IP burnt handling
- HCaptcha / anti-abuse bypass
- Multi-account quarantine logic

If you hit rate limits, slow the worker count down or add a paid plan.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import re
import sys
from pathlib import Path
from typing import Optional

import aiohttp
from dotenv import load_dotenv
from pydub import AudioSegment


API_BASE = "https://api.elevenlabs.io"
DEFAULT_MODEL = "eleven_multilingual_v2"
DEFAULT_CHUNK_SIZE = 1500          # chars per chunk (sentence-aware)
DEFAULT_WORKERS = 4
DEFAULT_REQUEST_TIMEOUT = 90.0     # seconds
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_S = 5.0


# ---------------------------------------------------------------------------
# Text chunking — sentence-aware, preserves abbreviations.
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
# ElevenLabs HTTP call (xi-api-key auth, no proxy, no Firebase, no captcha).
# ---------------------------------------------------------------------------

class TTSError(Exception):
    pass


class QuotaExhausted(TTSError):
    pass


class ContentBlocked(TTSError):
    pass


async def synth_chunk(
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
    """Single chunk → MP3 bytes.

    Retries on transient HTTP/network errors.  Maps 401→TTSError,
    402/403→QuotaExhausted, content moderation→ContentBlocked.
    """
    url = f"{API_BASE}/v1/text-to-speech/{voice_id}/stream"
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


# ---------------------------------------------------------------------------
# Parallel synth + concat.
# ---------------------------------------------------------------------------

async def synthesize(
    *,
    text: str,
    output_path: Path,
    api_key: str,
    voice_id: str,
    model: str,
    stability: float,
    similarity_boost: float,
    style: float,
    speed: float,
    chunk_size: int,
    workers: int,
    timeout: float,
    max_retries: int,
    backoff_s: float,
) -> None:
    chunks = split_text(text, chunk_size)
    if not chunks:
        raise ValueError("input text is empty after stripping")

    print(
        f"text   : {len(text)} chars → {len(chunks)} chunks "
        f"(<= {chunk_size} chars each)",
    )
    print(
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
                    audio = await synth_chunk(
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
    size_kb = output_path.stat().st_size / 1024
    print(f"\n✓ {output_path}  ({size_kb:,.1f} KB)")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Standalone ElevenLabs TTS (xi-api-key auth).",
    )
    ap.add_argument("input", help="path to .txt file with the script")
    ap.add_argument(
        "-o", "--output", default="output.mp3",
        help="output MP3 path (default: output.mp3)",
    )
    ap.add_argument(
        "--voice", default=os.getenv("VOICE_ID"),
        help="ElevenLabs voice_id (env: VOICE_ID)",
    )
    ap.add_argument(
        "--model", default=os.getenv("MODEL", DEFAULT_MODEL),
        help=f"model_id (default: {DEFAULT_MODEL})",
    )
    ap.add_argument("--stability", type=float, default=0.5)
    ap.add_argument("--similarity", type=float, default=0.75)
    ap.add_argument("--style", type=float, default=0.3)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument(
        "--chunk", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"chunk size in chars (default: {DEFAULT_CHUNK_SIZE})",
    )
    ap.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"parallel workers (default: {DEFAULT_WORKERS})",
    )
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
    return ap.parse_args()


def main() -> int:
    load_dotenv()  # picks up .env in CWD
    args = parse_args()

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print(
            "ERROR: ELEVENLABS_API_KEY not set.\n"
            "Put it in .env (see .env.example) or export it before running.",
            file=sys.stderr,
        )
        return 2

    if not args.voice:
        print(
            "ERROR: voice_id not provided.\n"
            "Pass --voice or set VOICE_ID in .env.",
            file=sys.stderr,
        )
        return 2

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        return 2
    text = in_path.read_text(encoding="utf-8")

    try:
        asyncio.run(synthesize(
            text=text,
            output_path=Path(args.output),
            api_key=api_key,
            voice_id=args.voice,
            model=args.model,
            stability=args.stability,
            similarity_boost=args.similarity,
            style=args.style,
            speed=args.speed,
            chunk_size=args.chunk,
            workers=args.workers,
            timeout=args.timeout,
            max_retries=args.retries,
            backoff_s=args.backoff,
        ))
    except (TTSError, QuotaExhausted, ContentBlocked) as e:
        print(f"\nFAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
