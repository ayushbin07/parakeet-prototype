import json
import queue
import re
import threading
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000

INLINE_RE = re.compile(
    r'(?P<code>`[^`\n]+`)|'
    r'(?P<bi_star>\*\*\*(?=\S)(.+?)(?<=\S)\*\*\*)|'
    r'(?P<bi_under>___(?=\S)(.+?)(?<=\S)___)|'
    r'(?P<b_star>\*\*(?=\S)(.+?)(?<=\S)\*\*)|'
    r'(?P<b_under>__(?=\S)(.+?)(?<=\S)__)|'
    r'(?P<i_star>\*(?=\S)(.+?)(?<=\S)\*)|'
    r'(?P<i_under>(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w))'
)


def parse_inline(text, current_tags=()):
    pos = 0
    spans = []
    for match in INLINE_RE.finditer(text):
        start, end = match.span()
        if start > pos:
            spans.append((text[pos:start], current_tags))

        g = match.lastgroup
        if g == 'code':
            spans.append((match.group('code')[1:-1], current_tags + ('inline_code',)))
        elif g in ('bi_star', 'bi_under'):
            inner = match.group(g)[3:-3]
            spans.extend(parse_inline(inner, current_tags + ('bold_italic',)))
        elif g in ('b_star', 'b_under'):
            inner = match.group(g)[2:-2]
            spans.extend(parse_inline(inner, current_tags + ('bold',)))
        elif g in ('i_star', 'i_under'):
            inner = match.group(g)[1:-1]
            spans.extend(parse_inline(inner, current_tags + ('italic',)))

        pos = end

    if pos < len(text):
        spans.append((text[pos:], current_tags))

    return spans


def resolve_tags(tags, block_prefix=None):
    is_bold = ('bold' in tags) or ('bold_italic' in tags)
    is_italic = ('italic' in tags) or ('bold_italic' in tags)
    is_code = ('inline_code' in tags)

    resolved = []
    if block_prefix:
        resolved.append(block_prefix)
        if block_prefix in ("h1", "h2", "h3"):
            if is_italic:
                resolved.append(f"{block_prefix}_italic")
            if is_code:
                resolved.append("inline_code")
            return tuple(resolved)

    if is_code:
        resolved.append("inline_code")
        if is_bold:
            resolved.append("bold")
        return tuple(resolved)

    if is_bold and is_italic:
        resolved.append("bold_italic")
    elif is_bold:
        resolved.append("bold")
    elif is_italic:
        resolved.append("italic")

    return tuple(resolved)


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

        self.setup_markdown_tags(self.answer_box)
        self.poll_queue()

    def setup_markdown_tags(self, widget):
        widget.tag_configure("h1", font=("TkDefaultFont", 16, "bold"), spacing1=8, spacing3=4)
        widget.tag_configure("h1_italic", font=("TkDefaultFont", 16, "bold italic"))
        widget.tag_configure("h2", font=("TkDefaultFont", 14, "bold"), spacing1=6, spacing3=3)
        widget.tag_configure("h2_italic", font=("TkDefaultFont", 14, "bold italic"))
        widget.tag_configure("h3", font=("TkDefaultFont", 12, "bold"), spacing1=4, spacing3=2)
        widget.tag_configure("h3_italic", font=("TkDefaultFont", 12, "bold italic"))
        widget.tag_configure("bold", font=("TkDefaultFont", 12, "bold"))
        widget.tag_configure("italic", font=("TkDefaultFont", 12, "italic"))
        widget.tag_configure("bold_italic", font=("TkDefaultFont", 12, "bold italic"))
        widget.tag_configure("inline_code", font=("TkFixedFont", 11), background="#ededed")
        widget.tag_configure("code_block", font=("TkFixedFont", 11), background="#f4f4f4", lmargin1=16, lmargin2=16)
        widget.tag_configure("bullet", lmargin1=16, lmargin2=34)
        widget.tag_configure("numbered", lmargin1=16, lmargin2=34)

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

    def render_answer(self, markdown_text):
        self.render_markdown(self.answer_box, markdown_text)

    def render_markdown(self, widget, markdown_text):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)

        lines = markdown_text.split("\n")
        in_code_block = False

        for idx, line in enumerate(lines):
            is_last = (idx == len(lines) - 1)
            newline = "" if is_last else "\n"

            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                widget.insert(tk.END, line + newline, ("code_block",))
                continue

            m_heading = re.match(r'^(#{1,6})\s+(.*)$', line)
            if m_heading:
                level = len(m_heading.group(1))
                h_text = m_heading.group(2)
                h_tag = "h1" if level == 1 else ("h2" if level == 2 else "h3")
                spans = parse_inline(h_text)
                if spans:
                    for span_text, tags in spans:
                        widget.insert(tk.END, span_text, resolve_tags(tags, h_tag))
                else:
                    widget.insert(tk.END, h_text, (h_tag,))
                if newline:
                    widget.insert(tk.END, newline, (h_tag,))
                continue

            m_bullet = re.match(r'^(\s*)([\*\-\+])\s+(.*)$', line)
            if m_bullet:
                indent = m_bullet.group(1)
                b_text = m_bullet.group(3)
                widget.insert(tk.END, indent + "• ", ("bullet",))
                spans = parse_inline(b_text)
                for span_text, tags in spans:
                    widget.insert(tk.END, span_text, resolve_tags(tags, "bullet"))
                if newline:
                    widget.insert(tk.END, newline, ("bullet",))
                continue

            m_num = re.match(r'^(\s*)(\d+)\.\s+(.*)$', line)
            if m_num:
                indent = m_num.group(1)
                num = m_num.group(2)
                n_text = m_num.group(3)
                widget.insert(tk.END, f"{indent}{num}. ", ("numbered",))
                spans = parse_inline(n_text)
                for span_text, tags in spans:
                    widget.insert(tk.END, span_text, resolve_tags(tags, "numbered"))
                if newline:
                    widget.insert(tk.END, newline, ("numbered",))
                continue

            spans = parse_inline(line)
            for span_text, tags in spans:
                widget.insert(tk.END, span_text, resolve_tags(tags))
            if newline:
                widget.insert(tk.END, newline)

        widget.see(tk.END)
        widget.config(state=tk.DISABLED)

    def poll_queue(self):
        latest_answer = None
        while not self.ui_queue.empty():
            action, args = self.ui_queue.get_nowait()
            if action == self.render_answer:
                latest_answer = args[0]
            else:
                if latest_answer is not None:
                    self.render_answer(latest_answer)
                    latest_answer = None
                action(*args)
        if latest_answer is not None:
            self.render_answer(latest_answer)
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
            accumulated_text = ""
            try:
                resp = requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={
                        "model": "llama3.2:latest",
                        "prompt": prompt,
                        "system": (
                            "Provide simple, brief, and direct answers in plain language, "
                            "with short sentences. Do not give a long explanation unless "
                            "the question asks for detail."
                        ),
                        "stream": True,
                    },
                    stream=True,
                    timeout=30,
                )
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        part = chunk.get("response", "")
                        if part:
                            accumulated_text += part
                            self.ui_queue.put((self.render_answer, (accumulated_text,)))
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

