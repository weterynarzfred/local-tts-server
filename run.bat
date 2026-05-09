@echo off

start "" "C:/Program Files/Waterfox/waterfox" http://localhost:3776
"C:/Program Files/Git/bin/bash.exe" -c "cd '/h/tts/chatterbox' && .venv/Scripts/uvicorn app:app --host 0.0.0.0 --port 3776"
