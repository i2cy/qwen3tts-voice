@echo off
rem start_cody_voice.bat - launch Cody Voice Service hidden (no console window) with system tray.
rem Use pythonw.exe so no black console pops up; --tray puts a status icon in the taskbar tray.
start "" "D:\ProgramData\anaconda3\envs\qwen3tts\pythonw.exe" "E:\Services\qwen3tts\qwen3tts_server.py" --port 8100 --tray
