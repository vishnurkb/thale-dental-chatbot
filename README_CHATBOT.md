# Thale Dental Chatbot - Local Setup

## Prerequisites
1. Install Ollama: https://ollama.com/download
2. Pull the model: `ollama pull llama3.2:3b`
3. `pip install -r requirements.txt` (or use the `D:/v` venv already set up: `D:/v/Scripts/pip.exe install -r requirements.txt`)

## Build the knowledge index (run once, re-run when site/ changes)
`D:/v/Scripts/python.exe -m chatbot.ingest.build_index`

## Run the backend
`D:/v/Scripts/python.exe -m uvicorn chatbot.api.main:app --reload --port 8001`

## Open the widget
Open `chatbot/widget/widget.html` directly in a browser (Ollama + backend must be running).

## Run tests
`D:/v/Scripts/python.exe -m pytest -v`
