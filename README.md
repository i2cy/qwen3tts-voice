# Cody Voice Service v1 — Qwen3-TTS-12Hz-1.7B-Base + fixed Fidget reference clone

Cody's real-time voice synthesis service. Runs on **i2pa** (RTX 5080), served
over the LAN as an OpenAI-compatible TTS endpoint, consumed by the OpenClaw
gateway (QQ voice messages, voice calls, ear channel) via an SSH tunnel.

## Architecture

```
[OpenClaw gateway / scripts]
        │  HTTP :8100 (SSH tunnel, -C compressed)
        ▼
[Go wrapper (optional, codyserver)  →  i2pa:8100 python server]
        ▼
[qwen3tts_server.py  (i2pa, GPU)  →  Qwen3TTSModel 12Hz-1.7B-Base, bf16, sdpa]
        ▼
[generate_voice_clone: Base + fixed ref fidget_ref_53_24k.wav]
```

- **Model**: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (ModelScope, 3.86GB bf16)
- **Ref audio**: `fidget_ref_53_24k.wav` — 4.8s continuous Fidget line
  ("It sure did! You know what I'm thinking? I'm thinking we find another one.")
  Fixed for ALL outputs so the timbre never drifts.
- **Attention**: `sdpa` (torch 2.13 built-in flash kernel on sm_120 — verified;
  external flash-attn not buildable on i2pa, no MSVC/CUDA toolkit).
- **Latency (hot, 5080)**: short "好的，爸爸" ≈ **1.4s**; full sentence ≈ 4.5–5s;
  RTF ≈ 1.3. Cold start (model load) ≈ 3s.
- **Idle unload**: model releases GPU+RAM after **3600s** without a call.

## API

```
POST /v1/audio/speech
  {"input": "...", "voice": "cody", "language": "Chinese|English|Auto",
   "instruct": "...", "max_tokens": 900, "format": "wav|flac"}
  → audio bytes (24kHz mono; wav or flac; +0.5s tail silence)

GET /health → {"ok": true, "model_loaded": bool, "idle_s": n}
```

- `format=flac` for efficient transport (≈50% of wav); the gateway decodes
  back to wav for QQ delivery.
- 0.5s trailing silence is appended server-side so QQ's end-fade never clips
  the last word.
- `instruct` is **only** forwarded to the model when non-empty (passing an
  empty string to `generate_voice_clone` degrades output — it has no instruct
  parameter).

## Deployment

- **i2pa** (Windows): `schtasks CodyVoiceServer` runs
  `D:\ProgramData\anaconda3\envs\qwen3tts\python.exe E:\Services\qwen3tts\qwen3tts_server.py --port 8100`,
  autostart at boot. Firewall rules `Qwen3TTS 8100` (in/out) already added.
- **Codyserver** (Linux): SSH tunnel for LAN access:
  `ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -C -i ~/.ssh/id_ed25519_i2proart -L 8100:127.0.0.1:8100 i2cy@192.168.11.179`
  (`-C` = compression for the flac/wav payload).
- Update code: `./deploy.sh --restart`.

## Notes / history

- 2026-08-31: indextts (RTF 3–4, non-streaming, TTFB≈total) **replaced** by
  Qwen3-TTS v1. First attempt at fine-tuning (`sft_12hz.py`, lr 2e-5, 10 epochs,
  122 clips) → EOS collapse (generation never stops, "ghost" audio). Re-run with
  lr 3e-6 × 4 epochs fixed EOS but timbre drifted per emotion ref; dad prefers
  the Base+fixed-ref clone — fine-tune parked as future work.
- 2026-08-31: server bug — writing the float32 array raw into a 16-bit WAV
  produced loud static; fixed via `soundfile` PCM_16 conversion.
- Todo: Go wrapper (single-binary gateway on codyserver), richer caching,
  streaming/real-time mode for QQ voice calls.
