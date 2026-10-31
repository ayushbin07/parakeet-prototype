import json
import queue
import threading
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Parakeet Prototype")

        self.audio_frames = []
        self.stream = None
        self.is_recording = False
        self.whisper_model = None
        self.ui_queue = queue.Queue()

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=8, pady=6)

        self.start_btn = tk.Button(btn_frame, text="Start", command=self.start_recording)
        self.start_btn.pack(side=tk.LEFT, padx=4)

        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        self.suggest_btn = tk.Button(
            btn_frame, text="Suggest Answer", command=self.suggest_answer, state=tk.DISABLED
        )
        self.suggest_btn.pack(side=tk.LEFT, padx=4)

        tk.Label(root, text="Transcript", anchor="w").pack(fill=tk.X, padx=8, pady=(4, 0))
        self.transcript_box = scrolledtext.ScrolledText(root, height=8, state=tk.DISABLED)
        self.transcript_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        tk.Label(root, text="Answer", anchor="w").pack(fill=tk.X, padx=8, pady=(4, 0))
        self.answer_box = scrolledtext.ScrolledText(root, height=12, state=tk.DISABLED)
        self.answer_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.poll_queue()

    def set_text(self, widget, text):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)

    def append_text(self, widget, text):
        widget.config(state=tk.NORMAL)
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.config(state=tk.DISABLED)

    def poll_queue(self):
        while not self.ui_queue.empty():
            action, args = self.ui_queue.get_nowait()
            action(*args)
        self.root.after(50, self.poll_queue)

    def start_recording(self):
        self.audio_frames = []
        self.is_recording = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.suggest_btn.config(state=tk.DISABLED)

        def record():
            try:
                def callback(indata, frames, time_info, status):
                    if self.is_recording:
                        self.audio_frames.append(indata.copy())

                with sd.InputStream(
                    samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
                ):
                    while self.is_recording:
                        sd.sleep(50)
            except Exception as e:
                self.ui_queue.put((self.on_mic_error, (str(e),)))

        threading.Thread(target=record, daemon=True).start()

    def on_mic_error(self, err_msg):
        self.set_text(self.answer_box, f"Microphone error: {err_msg}")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def stop_recording(self):
        self.is_recording = False
        self.stop_btn.config(state=tk.DISABLED)

        def transcribe():
            try:
                if not self.audio_frames:
                    text = ""
                else:
                    audio_data = np.concatenate(self.audio_frames, axis=0).flatten()
                    if self.whisper_model is None:
                        self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
                    segments, _ = self.whisper_model.transcribe(audio_data, language="en")
                    text = " ".join(s.text.strip() for s in segments)
                self.ui_queue.put((self.on_transcribe_done, (text,)))
            except Exception as e:
                self.ui_queue.put((self.set_text, (self.answer_box, f"Transcription error: {e}")))
                self.ui_queue.put((self.start_btn.config, ({"state": tk.NORMAL},)))

        threading.Thread(target=transcribe, daemon=True).start()

    def on_transcribe_done(self, text):
        self.set_text(self.transcript_box, text)
        self.start_btn.config(state=tk.NORMAL)
        self.suggest_btn.config(state=tk.NORMAL)

    def suggest_answer(self):
        self.suggest_btn.config(state=tk.DISABLED)
        self.set_text(self.answer_box, "")
        prompt = self.transcript_box.get("1.0", tk.END).strip()

        def stream_ollama():
            try:
                resp = requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": "llama3.2:latest", "prompt": prompt, "stream": True},
                    stream=True,
                    timeout=30,
                )
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        part = chunk.get("response", "")
                        self.ui_queue.put((self.append_text, (self.answer_box, part)))
            except Exception as e:
                self.ui_queue.put((self.set_text, (self.answer_box, f"Ollama error: {e}")))
            finally:
                self.ui_queue.put((self.suggest_btn.config, ({"state": tk.NORMAL},)))

        threading.Thread(target=stream_ollama, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
