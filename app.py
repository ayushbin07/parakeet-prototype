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
        self.root.geometry("750x650")

        self.audio_frames = []
        self.is_recording = False
        self.whisper_model = None
        self.ui_queue = queue.Queue()

        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.status_var = tk.StringVar(value="Idle")
        self.status_label = tk.Label(main_frame, textvariable=self.status_var, anchor="w", font=("TkDefaultFont", 11))
        self.status_label.pack(fill=tk.X, pady=(0, 8))

        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 8))

        self.start_btn = tk.Button(
            btn_frame, text="Start", command=self.start_recording, font=("TkDefaultFont", 11), padx=12, pady=4
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_btn = tk.Button(
            btn_frame, text="Stop", command=self.stop_recording, state=tk.DISABLED, font=("TkDefaultFont", 11), padx=12, pady=4
        )
        self.stop_btn.pack(side=tk.LEFT)

        tk.Label(main_frame, text="Transcript", anchor="w", font=("TkDefaultFont", 11, "bold")).pack(fill=tk.X, pady=(4, 2))
        self.transcript_box = scrolledtext.ScrolledText(main_frame, height=7, state=tk.DISABLED, font=("TkDefaultFont", 12))
        self.transcript_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        tk.Label(main_frame, text="Answer", anchor="w", font=("TkDefaultFont", 11, "bold")).pack(fill=tk.X, pady=(4, 2))
        self.answer_box = scrolledtext.ScrolledText(main_frame, height=11, state=tk.DISABLED, font=("TkDefaultFont", 12))
        self.answer_box.pack(fill=tk.BOTH, expand=True)

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
        self.status_var.set("Recording")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

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
        self.status_var.set("Idle")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def stop_recording(self):
        self.is_recording = False
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Transcribing")

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
                self.ui_queue.put((self.on_transcribe_error, (str(e),)))

        threading.Thread(target=transcribe, daemon=True).start()

    def on_transcribe_error(self, err_msg):
        self.set_text(self.answer_box, f"Transcription error: {err_msg}")
        self.status_var.set("Idle")
        self.start_btn.config(state=tk.NORMAL)

    def on_transcribe_done(self, text):
        self.set_text(self.transcript_box, text)
        if not text.strip():
            self.set_text(self.answer_box, "No speech detected, try again")
            self.status_var.set("Idle")
            self.start_btn.config(state=tk.NORMAL)
            return

        self.start_answer_stream(text.strip())

    def start_answer_stream(self, prompt):
        self.status_var.set("Answering")
        self.set_text(self.answer_box, "")

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
                self.ui_queue.put((self.status_var.set, ("Done",)))
            except Exception as e:
                self.ui_queue.put((self.set_text, (self.answer_box, f"Ollama error: {e}")))
                self.ui_queue.put((self.status_var.set, ("Idle",)))
            finally:
                self.ui_queue.put((self.start_btn.config, ({"state": tk.NORMAL},)))

        threading.Thread(target=stream_ollama, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
