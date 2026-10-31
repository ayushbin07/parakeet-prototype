# Parakeet Prototype

A small local desktop assistant, inspired by Parakeet AI.

Speak a question, see the transcription, then ask for a suggested answer. Everything runs locally. No audio or transcript is sent to a cloud service.

## What it does

- Opens a simple desktop window with Start and Stop buttons.
- Captures microphone audio while recording is on.
- Displays the transcription of what was said.
- When Stop is pressed, after transcription it automatically sends the transcript to a local language model and displays the answer progressively as it is generated.

## Stack

- Python with Tkinter for the desktop window. This runs on Windows, Mac, and Linux from one codebase.
- faster-whisper for local speech to text.
- Ollama for the local language model, over local HTTP with streaming.
- sounddevice for microphone capture.

## Setup

1. Install Ollama and start it:
   - `ollama serve`
   - `ollama pull llama3.2`
2. Install Python dependencies:
   - `pip install -r requirements.txt`
3. Run the app:
   - `python3 app.py`

The default model is `llama3.2`. Change it in the app settings if you want a different local Ollama model.

## Project skills

Skills used for this build live in `skills/`:

- ponytail: smallest complete change, no extra features.
- karpathy-guidelines: simple code with verifiable checks.
- frontend-design: used lightly, for a clean window layout.

## Status

Prototype in progress. The window, microphone capture, transcription, and streaming answer are built in small steps, each verified before the next one starts.
# parakeet-prototype
# parakeet-prototype
# parakeet-prototype
# parakeet-prototype
