@echo off
start http://localhost:3776
.venv\Scripts\uvicorn app:app --host 0.0.0.0 --port 3776
