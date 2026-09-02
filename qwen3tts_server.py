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
REF_AUDIO_EN = os.path.join(BASE_DIR, "fidget_ref_53_24k.wav")
REF_TEXT_EN = "It sure did! You know what I'm thinking? I'm thinking we find another one."
# dad-picked Chinese reference (Sep 03 2026): the warm reply he loved from the
# orangeBox ("嘿，我在呢，dad…"), pitch-preserved speed-up x1.2 so Chinese
# pacing feels livelier (dad listened 1.0/1.1/1.2/1.3/1.4x, picked 1.2x).
# Chinese synthesis uses this clip; English keeps the original Fidget ref.
REF_AUDIO_ZH = os.path.join(BASE_DIR, "zh_ref_dad_12x_20260903.wav")
REF_TEXT_ZH = "嘿，我在呢，dad。大半夜的还惦记着试试这个橙色小盒子呀——我听得很清楚哦。怎么啦？"
DEFAULT_LANG = "Auto"
IDLE_UNLOAD_S = 3600  # 1h no calls -> release GPU + RAM
TAIL_PAD_S = 0.5       # pad silence so QQ's end-fade doesn't clip the last word
DEFAULT_SEED = 42      # English seed (dad-picked Aug 31). Chinese seed: 21
                       # (dad-picked Sep 03, no leading space — livelier pacing).
DEFAULT_SEED_ZH = 21
DEFAULT_LS_EN = True   # English: leading space paces naturally (dad Aug 31)
DEFAULT_LS_ZH = False  # Chinese: no leading space — dad-picked Sep 03

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
    """Lazy-loads the model + clone prompt; unloads after IDLE_UNLOAD_S."""

    def __init__(self):
        self._model = None
        self._prompts = {}
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
        log("model loaded in %.2fs" % (time.time() - t0))
        # build BOTH voice-clone prompts ONCE and reuse for every request:
        # create_voice_clone_prompt re-encodes the ref audio (~0.9s) each call.
        self._prompts = {}
        for key, ref_audio, ref_text, label in (
            ("en", REF_AUDIO_EN, REF_TEXT_EN, "fidget_en"),
            ("zh", REF_AUDIO_ZH, REF_TEXT_ZH, "zh_dad_20260903"),
        ):
            t1 = time.time()
            self._prompts[key] = m.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
            log("clone prompt[%s] built in %.2fs (cached)" % (label, time.time() - t1))
        return m

    def get(self):
        with self._lock:
            if self._model is None:
                self._model = self._load()
            self._last_use = time.time()
            if self._idle_timer:
                self._idle_timer.cancel()
            return self._model

    def get_prompt(self, lang_key="en"):
        self.get()
        return self._prompts.get(lang_key) or self._prompts["en"]

    def _unload(self):
        with self._lock:
            if self._model is None:
                return
            import torch

            torch.cuda.empty_cache()
            self._model = None
            self._prompts = {}
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


def _lang_key(text, language):
    """Choose the voice reference: Chinese ref for Chinese requests / CJK-heavy
    text, English ref otherwise (Auto falls back to content detection)."""
    if language and language.lower().startswith("zh"):
        return "zh"
    if language and language.lower() == "english":
        return "en"
    # Auto / None: detect CJK presence
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return "zh" if cjk >= max(2, len(text) // 20) else "en"


def synth(text, language, instruct, max_tokens, seed, leading_space=True):
    model = host.get()
    lang_key = _lang_key(text, language)
    prompt = host.get_prompt(lang_key)
    try:
        # fixed seed = stable pacing/timbre (dad-picked 42, Aug 31).
        # seed=0 means random. Callers may override per request.
        if seed:
            import torch

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        kwargs = dict(
            # leading space slows/paces delivery more naturally (dad-picked,
            # Aug 31: "Wow!!..." exclaim with leading space beats raw start).
            # Chinese replies may prefer no leading space (faster, more natural
            # attack) — controllable per request via body["leading_space"].
            text=(" " + text) if leading_space else text,
            language=language if language and language != "Auto" else None,
            voice_clone_prompt=prompt,
            max_new_tokens=max_tokens,
            # explicit empty instruct ("" != absent) gives a calmer, cuter
            # pacing — dad-picked (Aug 31). Callers may override with real
            # emotion/pacing instructions, forwarded verbatim.
            instruct=instruct or "",
        )
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
                "refs": {
                    "en": os.path.basename(REF_AUDIO_EN),
                    "zh": os.path.basename(REF_AUDIO_ZH),
                },
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
        is_zh = _lang_key(text, language) == "zh"
        instruct = body.get("instruct") or body.get("emo_text") or ""
        # language-aware defaults: en=seed42+leading space, zh=seed21+no space.
        # explicit body values always win.
        raw_seed = body.get("seed", DEFAULT_SEED_ZH if is_zh else DEFAULT_SEED)
        seed = int(raw_seed) if raw_seed not in (None, "", 0) else 0
        max_tokens = int(body.get("max_tokens", 900))
        ls_default = DEFAULT_LS_ZH if is_zh else DEFAULT_LS_EN
        leading_space = body.get("leading_space", ls_default)
        if not isinstance(leading_space, bool):
            leading_space = str(leading_space).lower() not in ("false", "0", "no")

        t0 = time.time()
        try:
            wav, sr = synth(text, language, instruct, max_tokens, seed, leading_space)
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


def start_tray(port):
    """System-tray icon (Windows). Run with `pythonw.exe --tray` so no console
    window appears; the tray offers status/health, log file, and quit."""
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        log("--tray requested but pystray/Pillow missing (pip install pystray pillow)")
        return

    img = Image.new("RGB", (64, 64), (20, 20, 20))
    d = ImageDraw.Draw(img)
    d.ellipse((10, 10, 54, 54), fill=(120, 180, 255))
    d.polygon([(22, 40), (22, 24), (42, 32)], fill=(255, 255, 255))

    def status_text():
        if host.loaded():
            idle = host.idle_s()
            return "model loaded (idle %ds)" % idle
        return "model unloaded"

    def on_status(_icon, _item):
        import webbrowser
        webbrowser.open("http://127.0.0.1:%d/health" % port)

    def on_log(_icon, _item):
        import os
        try:
            os.startfile(os.path.join(BASE_DIR, "qwen3tts_server.log"))
        except Exception as e:
            log("open log failed: %s" % e)

    def on_quit(_icon, _item):
        _icon.stop()
        log("tray quit requested, exiting")
        os._exit(0)

    icon = pystray.Icon(
        "cody-voice",
        img,
        "Cody Voice Service v1",
        menu=pystray.Menu(
            pystray.MenuItem(lambda item: "Voice: " + status_text(), action=None, enabled=False),
            pystray.MenuItem("Status / health", on_status),
            pystray.MenuItem("Open log file", on_log),
            pystray.MenuItem("Quit service", on_quit),
        ),
    )
    threading.Thread(target=icon.run, daemon=True).start()
    log("tray icon started")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--tray", action="store_true", help="show system-tray icon (Windows, pythonw)")
    args = ap.parse_args()
    if args.tray:
        start_tray(args.port)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log("qwen3tts voice server v1 listening on %s:%d (idle unload %ds)" % (args.host, args.port, IDLE_UNLOAD_S))
    srv.serve_forever()


if __name__ == "__main__":
    main()
