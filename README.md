# ElevenLabs TTS — minimal CLI

A self-contained script that turns a `.txt` script into an `.mp3` via ElevenLabs.

It does **only** the things you can share publicly:

- Authenticates with a paid `xi-api-key`
- Splits long scripts into sentence-aware chunks
- Calls `/v1/text-to-speech/{voice_id}/stream` in parallel
- Stitches MP3 chunks back together with `pydub`

It deliberately leaves out everything that depends on private accounts or platform-evasion infrastructure (Firebase free-tier login, proxy ports, IP-burnt rotation, captcha bypass, account quarantine). If you hit `429`, lower `--workers` or upgrade your ElevenLabs plan.

## Install

Requires **Python 3.10+** and **ffmpeg** (used by `pydub` for MP3 stitching).

```bash
# macOS
brew install ffmpeg
# Ubuntu / Debian
sudo apt install ffmpeg

# Project deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Copy `.env.example` → `.env` and fill in your key + voice id:

```bash
cp .env.example .env
$EDITOR .env
```

```ini
ELEVENLABS_API_KEY=sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
VOICE_ID=21m00Tcm4TlvDq8ikWAM
MODEL=eleven_multilingual_v2
```

- Get the API key: <https://elevenlabs.io/app/settings/api-keys>
- Browse voices: <https://elevenlabs.io/app/voice-library>

## Run

```bash
# Use defaults from .env
python tts.py script.txt -o my_voice.mp3

# Override per-run
python tts.py script.txt \
    -o my_voice.mp3 \
    --voice 21m00Tcm4TlvDq8ikWAM \
    --model eleven_multilingual_v2 \
    --stability 0.5 --similarity 0.75 --style 0.3 --speed 1.0 \
    --chunk 1500 --workers 4
```

### Useful flags

| Flag           | Default | Meaning                                                 |
| -------------- | ------- | ------------------------------------------------------- |
| `--voice`      | `$VOICE_ID` | ElevenLabs voice id                                 |
| `--model`      | `eleven_multilingual_v2` | also: `eleven_flash_v2_5`, `eleven_turbo_v2_5` |
| `--stability`  | `0.5`   | 0.0 – 1.0 (lower = more expressive)                     |
| `--similarity` | `0.75`  | 0.0 – 1.0 (higher = closer to the reference voice)      |
| `--style`      | `0.3`   | 0.0 – 1.0 (style exaggeration)                          |
| `--speed`      | `1.0`   | 0.7 – 1.2 typical                                        |
| `--chunk`      | `1500`  | chars per chunk (lower = smaller HTTP requests)         |
| `--workers`    | `4`     | parallel HTTP requests (start at 4, raise if no 429s)   |
| `--timeout`    | `90`    | per-request timeout in seconds                          |
| `--retries`    | `5`     | retries per chunk on 429/5xx/network errors             |
| `--backoff`    | `5.0`   | linear backoff base in seconds (`backoff * attempt`)    |

## Exit codes

- `0` — success, MP3 written
- `1` — synthesis failed (auth / quota / chunks gave up)
- `2` — bad input (missing `.env` value, missing input file, bad CLI arg)

## Pricing & limits

This script makes **paid API calls**. Each character billed at your plan's rate. Concurrency is limited per plan — start with 4 workers and raise only if your plan supports more.

Free-tier keys won't work for any non-trivial workload — ElevenLabs blocks free-tier `xi-api-key` requests for some voices and slaps `429` aggressively. Buy a Starter plan or use the public free voices that are explicitly marked as such in the voice library.
