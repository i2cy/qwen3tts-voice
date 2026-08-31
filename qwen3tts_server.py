#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen3tts_server.py — Cody's voice service v1 (Qwen3-TTS-12Hz-1.7B-Base + fixed Fidget ref)
Location: E:\\Services\\qwen3tts\\  (i2pa, RTX 5080)
Pipeline: Qwen3TTSModel (Base-1.7B, bf16, sdpa) + generate_voice_clone with a
          fixed reference clip (fidget_ref_53_24k.wav) so the timbre NEVER drifts.
Latency:  rtf ~1.3 hot (5080).  Model unloads after IDLE_UNLOAD_S (1h) of no calls.
API (OpenAI-compatible subset):
  POST /v1/audio/speech  {"input": "...", "voice": "cody", "language": "Chinese"|"English"|"Auto", "instruct": "..."}
      -> wav bytes (24000 Hz mono s16)
  GET  /health           -> {"ok": true, "model_loaded": bool, "idle_s": n}
Run:    D:\\ProgramData\\anaconda3\\envs\\qwen3tts\\python.exe qwen3tts_server.py [--port 8100]
Autostart: schtasks CodyVoiceServer (see cody_voice_service.md)
"""
import argparse
import gc
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Base-1.7B")
REF_AUDIO = os.path.join(BASE_DIR, "fidget_ref_53_24k.wav")
REF_TEXT = "It sure did! You know what I'm thinking? I'm thinking we find another one."
DEFAULT_LANG = "Auto"
IDLE_UNLOAD_S = 3600  # 1h no calls -> release GPU + RAM
TAIL_PAD_S = 0.5       # pad silence so QQ's end-fade doesn't clip the last word

_log_lock = threading.Lock()


def log(msg):
    with _log_lock:
        line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
        print(line, flush=True)
        try:
            with open(os.path.join(BASE_DIR, "qwen3tts_server.log"), "a") as f:
                f.write(line + "\n")
        except Exception:
            pass


class ModelHost:
    """Lazy-loads the model; unloads after IDLE_UNLOAD_S of inactivity."""

    def __init__(self):
        self._model = None
        self._last_use = 0.0
        self._lock = threading.Lock()
        self._idle_timer = None

    def _load(self):
        import torch
        from qwen_tts import Qwen3TTSModel

        t0 = time.time()
        m = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        log("model loaded in %.2fs (ref=%s)" % (time.time() - t0, os.path.basename(REF_AUDIO)))
        return m

    def get(self):
        with self._lock:
            if self._model is None:
                self._model = self._load()
            self._last_use = time.time()
            if self._idle_timer:
                self._idle_timer.cancel()
            return self._model

    def _unload(self):
        with self._lock:
            if self._model is None:
                return
            import torch

            torch.cuda.empty_cache()
            self._model = None
            gc.collect()
            torch.cuda.empty_cache()
            log("model unloaded (idle > %ds), GPU + RAM released" % IDLE_UNLOAD_S)

    def touch_idle_timer(self):
        with self._lock:
            if self._idle_timer:
                self._idle_timer.cancel()
            self._idle_timer = threading.Timer(IDLE_UNLOAD_S, self._unload)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def idle_s(self):
        if self._model is None:
            return -1
        return int(time.time() - self._last_use)

    def loaded(self):
        return self._model is not None


host = ModelHost()


def synth(text, language, instruct, max_tokens):
    model = host.get()
    try:
        kwargs = dict(
            text=text,
            language=language if language and language != "Auto" else None,
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            max_new_tokens=max_tokens,
        )
        # generate_voice_clone has NO instruct/emo_text param — passing empty
        # strings as kwargs degrades the clone. Only forward non-empty extras.
        if instruct:
            kwargs["instruct"] = instruct
        w, sr = model.generate_voice_clone(**kwargs)
        return w[0], sr
    finally:
        host.touch_idle_timer()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence
        pass

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n > 0 else b""

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            return self._json(200, {
                "ok": True,
                "model_loaded": host.loaded(),
                "idle_s": host.idle_s(),
                "ref": os.path.basename(REF_AUDIO),
            })
        self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path != "/v1/audio/speech":
            return self._json(404, {"error": "not found"})
        try:
            body = json.loads(self._read_body().decode() or "{}")
        except Exception as e:
            return self._json(400, {"error": "bad json: %s" % e})

        text = body.get("input") or body.get("text")
        if not text:
            return self._json(400, {"error": "input required"})
        language = body.get("language", DEFAULT_LANG)
        instruct = body.get("instruct") or body.get("emo_text") or ""
        max_tokens = int(body.get("max_tokens", 900))

        t0 = time.time()
        try:
            wav, sr = synth(text, language, instruct, max_tokens)
        except Exception as e:
            import traceback

            traceback.print_exc()
            return self._json(500, {"error": str(e)})

        # soundfile handles float32 -> PCM_16 conversion (raw .tobytes() of a
        # float array written into a 16-bit WAV = static/noise)
        import io
        import numpy as np
        import soundfile as sf

        # pad trailing silence so the receiving side's fade-out (QQ) never
        # clips the final word
        if TAIL_PAD_S > 0:
            pad = np.zeros(int(sr * TAIL_PAD_S), dtype=wav.dtype)
            wav = np.concatenate([wav, pad])

        fmt = (body.get("format") or "wav").lower()
        if fmt == "flac":
            buf = io.BytesIO()
            sf.write(buf, wav, sr, format="FLAC", subtype="PCM_16")
            data = buf.getvalue()
            ctype = "audio/flac"
        else:
            buf = io.BytesIO()
            sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
            data = buf.getvalue()
            ctype = "audio/wav"
        data = buf.getvalue()
        dt = time.time() - t0
        log("synth lang=%s len=%d chars dt=%.2fs audio=%.1fs" % (language, len(text), dt, len(wav) / sr))

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Gen-Time-Ms", str(int(dt * 1000)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log("qwen3tts voice server v1 listening on %s:%d (idle unload %ds)" % (args.host, args.port, IDLE_UNLOAD_S))
    srv.serve_forever()


if __name__ == "__main__":
    main()
