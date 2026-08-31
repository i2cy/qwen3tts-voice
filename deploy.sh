#!/usr/bin/env bash
# deploy.sh — push voice-service code to i2pa (E:\Services\qwen3tts) and restart the schtasks service.
# Usage: ./deploy.sh [--restart]
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
HOST="i2cy@192.168.11.179"
DST='E:/Services/qwen3tts'

echo "== scp code -> i2pa =="
scp -i ~/.ssh/id_ed25519_i2proart -o ConnectTimeout=10 -o ServerAliveInterval=15 \
    "$SRC/qwen3tts_server.py" "$HOST:$DST/"

if [ "$1" = "--restart" ]; then
    echo "== restarting CodyVoiceServer =="
    ssh -i ~/.ssh/id_ed25519_i2proart "$HOST" \
        'powershell -NoProfile -Command "schtasks /End /TN CodyVoiceServer 2>&1 | Out-Null; Start-Sleep 2; schtasks /Run /TN CodyVoiceServer"'
    echo "== done =="
fi