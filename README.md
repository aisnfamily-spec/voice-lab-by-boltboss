# Voice Lab

A no-frills TTS CLI: feed it a `.txt`, get an `.mp3`. Two engines:

| Engine | Cost | Quality | Internet | Setup |
| --- | --- | --- | --- | --- |
| **ElevenLabs** | **paid** API key | premium | required | 30 sec — register, copy key |
| **Piper** | **free** — runs locally | decent (varies by voice) | not needed after model download | 2 min — pip install + download model |

Both engines share the same chunker, parallelism story (Piper is sequential for safety), and stitching pipeline. Only the inner HTTP/inference call differs.

What's intentionally **not** here: Firebase free-tier login, proxy rotation, captcha bypass, multi-account quarantine. Just two clean, public-friendly paths.

## Install

Requires **Python 3.10+** and **ffmpeg** (used by `pydub` for audio stitching).

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

If you want the **Piper** engine, also install it (and download a voice — see below):

```bash
pip install piper-tts
```

## Configure

```bash
cp .env.example .env
$EDITOR .env
```

For ElevenLabs you need:

```ini
ENGINE=elevenlabs
ELEVENLABS_API_KEY=sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
VOICE_ID=21m00Tcm4TlvDq8ikWAM
MODEL=eleven_multilingual_v2
```

For Piper, the `.env` is optional — every value can be passed on the CLI.

## Run

### ElevenLabs (paid, premium quality)

```bash
python tts.py script.txt -o my_voice.mp3
```

Override per-run:

```bash
python tts.py script.txt -o out.mp3 \
    --voice 21m00Tcm4TlvDq8ikWAM \
    --model eleven_multilingual_v2 \
    --stability 0.5 --similarity 0.75 --style 0.3 --speed 1.0 \
    --workers 4
```

### Piper (free, offline)

Download a voice once. Catalog: <https://github.com/rhasspy/piper/blob/master/VOICES.md>

```bash
mkdir -p voices && cd voices
# English (US) — Amy, medium quality
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
cd ..
```

Each voice = `.onnx` (model) + `.onnx.json` (config). Both must sit in the same folder.

Then synthesize:

```bash
python tts.py script.txt -o out.mp3 \
    --engine piper \
    --piper-model voices/en_US-amy-medium.onnx \
    --piper-length-scale 1.0
```

Increase `--piper-length-scale` to slow down (e.g. `1.2`), decrease to speed up (e.g. `0.9`).

## Flags

| Flag                  | Default                    | Engine        | Meaning                                                   |
| --------------------- | -------------------------- | ------------- | --------------------------------------------------------- |
| `--engine`            | `elevenlabs`               | both          | `elevenlabs` or `piper`                                   |
| `-o, --output`        | `output.mp3`               | both          | output MP3 path                                           |
| `--chunk`             | `1500`                     | both          | chars per sentence-aware chunk                            |
| `--workers`           | `4`                        | ElevenLabs    | parallel HTTP requests                                    |
| `--voice`             | `$VOICE_ID`                | ElevenLabs    | voice id                                                  |
| `--model`             | `eleven_multilingual_v2`   | ElevenLabs    | model id                                                  |
| `--stability`         | `0.5`                      | ElevenLabs    | 0.0 – 1.0                                                  |
| `--similarity`        | `0.75`                     | ElevenLabs    | 0.0 – 1.0                                                  |
| `--style`             | `0.3`                      | ElevenLabs    | 0.0 – 1.0                                                  |
| `--speed`             | `1.0`                      | ElevenLabs    | 0.7 – 1.2 typical                                         |
| `--timeout`           | `90`                       | ElevenLabs    | per-request timeout (s)                                   |
| `--retries`           | `5`                        | ElevenLabs    | retries per chunk on 429/5xx/network                      |
| `--backoff`           | `5.0`                      | ElevenLabs    | linear backoff base (`backoff * attempt`)                 |
| `--piper-model`       | —                          | Piper         | path to `.onnx` model                                     |
| `--piper-length-scale`| `1.0`                      | Piper         | speak rate (>1 slower, <1 faster)                         |

## Exit codes

- `0` — success, MP3 written
- `1` — synthesis failed (auth / quota / chunks gave up)
- `2` — bad input (missing `.env` value, missing input file, bad CLI arg)

## Cost / limits

**ElevenLabs free tier** gives ~10 000 chars/month — enough to test, not for production. Starter (`$5/month`) raises that to 30 000, Creator (`$22/month`) to 100 000. If you hit `429`, drop `--workers` first; only then upgrade your plan.

**Piper** is FOSS (MIT) and runs entirely on your machine. There are no per-character fees, no rate limits, no internet. Quality of `medium` voices is reasonable for most narration; `high` quality voices sound noticeably better but cost more CPU. `x_low` voices are tiny but robotic.

## Picking between the two

- Need ad-quality narration? → ElevenLabs.
- Bulk processing, no budget, or paranoid about sending text to a third party? → Piper.
- Best of both: prototype with Piper, ship final cuts with ElevenLabs.
