import os
import sys
import queue
import threading
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# When frozen as .exe, add bundled ffmpeg to PATH
if getattr(sys, 'frozen', False):
    _bundle_dir = sys._MEIPASS
    os.environ['PATH'] = _bundle_dir + os.pathsep + os.environ.get('PATH', '')

LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]
AUDIO_FILETYPES = [("Audio/Video files", "*.wav *.mp3 *.mp4 *.avi *.mov *.mkv *.flac *.ogg *.m4a *.webm"),
                   ("All files", "*.*")]
TEXT_FILETYPES = [("Text files", "*.txt *.md"), ("All files", "*.*")]


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))


class VoicePairRow:
    def __init__(self, parent, index, remove_callback):
        self.voice_path = None
        self.frame = ttk.Frame(parent, relief="groove", padding=6)
        self.frame.pack(fill="x", pady=3, padx=4)

        # Row header
        ttk.Label(self.frame, text=f"#{index}", width=3, anchor="center").grid(row=0, column=0, rowspan=2, padx=(0, 6))

        # Text area
        ttk.Label(self.frame, text="Text:").grid(row=0, column=1, sticky="w")
        self.text_widget = tk.Text(self.frame, height=3, width=45, wrap="word", font=("Segoe UI", 9))
        self.text_widget.grid(row=1, column=1, padx=(0, 6), sticky="ew")

        # Text file load button
        ttk.Button(self.frame, text="Load .txt", width=8,
                   command=self._load_txt).grid(row=1, column=2, padx=(0, 6), sticky="n")

        # Voice section
        voice_frame = ttk.Frame(self.frame)
        voice_frame.grid(row=0, column=3, rowspan=2, padx=(0, 6), sticky="ns")
        ttk.Label(voice_frame, text="Voice file:").pack(anchor="w")
        self.voice_label = ttk.Label(voice_frame, text="(none)", foreground="gray", width=22, anchor="w")
        self.voice_label.pack(anchor="w")
        ttk.Button(voice_frame, text="Browse...", command=self._browse_voice).pack(anchor="w", pady=(4, 0))

        # Language dropdown
        lang_frame = ttk.Frame(self.frame)
        lang_frame.grid(row=0, column=4, rowspan=2, padx=(0, 6), sticky="ns")
        ttk.Label(lang_frame, text="Lang:").pack(anchor="w")
        self.lang_var = tk.StringVar(value="en")
        ttk.OptionMenu(lang_frame, self.lang_var, "en", *LANGUAGES).pack(anchor="w")

        # Remove button
        ttk.Button(self.frame, text="✕", width=3,
                   command=lambda: remove_callback(self)).grid(row=0, column=5, rowspan=2, padx=(0, 2))

        self.frame.columnconfigure(1, weight=1)

    def _load_txt(self):
        path = filedialog.askopenfilename(filetypes=TEXT_FILETYPES, title="Select text file")
        if path:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.text_widget.delete("1.0", "end")
            self.text_widget.insert("1.0", content)

    def _browse_voice(self):
        path = filedialog.askopenfilename(filetypes=AUDIO_FILETYPES, title="Select reference voice")
        if path:
            self.voice_path = path
            name = os.path.basename(path)
            display = name if len(name) <= 24 else name[:21] + "..."
            self.voice_label.config(text=display, foreground="black")

    def get_text(self):
        return self.text_widget.get("1.0", "end").strip()

    def validate(self):
        if not self.get_text():
            return "Text is empty."
        if not self.voice_path:
            return "No voice file selected."
        return None

    def destroy(self):
        self.frame.destroy()


class StdoutRedirector:
    """Redirect print() output to a queue consumed by the GUI log."""
    def __init__(self, q):
        self._queue = q
        self._orig = sys.stdout

    def write(self, text):
        if text.strip():
            self._queue.put(text.rstrip())
        self._orig.write(text)

    def flush(self):
        self._orig.flush()


class VoiceCloneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VoiceClone Studio")
        self.geometry("820x700")
        self.minsize(700, 500)
        self.resizable(True, True)

        self._rows: list[VoicePairRow] = []
        self._cloner = None
        self._model_ready = False
        self._log_queue = queue.Queue()

        self._build_ui()
        self._poll_log()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Header
        ttk.Label(self, text="VoiceClone Studio", font=("Segoe UI", 16, "bold")).pack(**pad, anchor="w")
        ttk.Separator(self).pack(fill="x", padx=10)

        # Voice pairs area
        pairs_label = ttk.Label(self, text="Voice Pairs", font=("Segoe UI", 10, "bold"))
        pairs_label.pack(anchor="w", padx=10, pady=(8, 2))

        self._scroll_frame = ScrollableFrame(self)
        self._scroll_frame.pack(fill="both", expand=True, padx=10)
        self._pairs_container = self._scroll_frame.inner

        ttk.Button(self, text="+ Add Voice Pair", command=self._add_row).pack(anchor="w", **pad)
        ttk.Separator(self).pack(fill="x", padx=10)

        # Settings row
        settings = ttk.Frame(self)
        settings.pack(fill="x", **pad)

        ttk.Label(settings, text="Output folder:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self._output_var = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "VoiceClone_output"))
        ttk.Entry(settings, textvariable=self._output_var, width=40).grid(row=0, column=1, sticky="ew")
        ttk.Button(settings, text="Browse...", command=self._browse_output).grid(row=0, column=2, padx=(6, 20))

        ttk.Label(settings, text="Silence (s):").grid(row=0, column=3, sticky="w", padx=(0, 4))
        self._silence_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(settings, textvariable=self._silence_var, from_=0.0, to=5.0,
                    increment=0.1, width=6, format="%.1f").grid(row=0, column=4)
        settings.columnconfigure(1, weight=1)

        # Voice settings sliders
        ttk.Label(self, text="Voice Settings", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(6, 2))
        vs = ttk.Frame(self)
        vs.pack(fill="x", padx=10, pady=(0, 4))

        def _slider_row(parent, row, label, var, from_, to, default, fmt="{:.2f}"):
            ttk.Label(parent, text=label, width=18, anchor="w").grid(row=row, column=0, sticky="w")
            val_lbl = ttk.Label(parent, text=fmt.format(default), width=5, anchor="e")
            val_lbl.grid(row=row, column=2, padx=(4, 16))
            def _on_change(v, lbl=val_lbl, f=fmt): lbl.config(text=f.format(float(v)))
            sl = ttk.Scale(parent, variable=var, from_=from_, to=to, orient="horizontal",
                           length=200, command=_on_change)
            sl.grid(row=row, column=1, sticky="ew")
            return sl

        self._speed_var       = tk.DoubleVar(value=1.0)
        self._temperature_var = tk.DoubleVar(value=0.85)
        self._top_p_var       = tk.DoubleVar(value=0.85)
        self._rep_pen_var     = tk.DoubleVar(value=5.0)

        _slider_row(vs, 0, "Speed",               self._speed_var,       0.5,  2.0,  1.0)
        _slider_row(vs, 1, "Variability (Temp)",  self._temperature_var, 0.01, 1.0,  0.85)
        _slider_row(vs, 2, "Similarity (Top-P)",  self._top_p_var,       0.01, 1.0,  0.85)
        _slider_row(vs, 3, "Repetition Penalty",  self._rep_pen_var,     1.0,  10.0, 5.0)
        vs.columnconfigure(1, weight=1)

        # Generate button
        self._gen_btn = ttk.Button(self, text="Generate", command=self._generate)
        self._gen_btn.pack(**pad)

        # Log area
        ttk.Label(self, text="Log", font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10)
        self._log = scrolledtext.ScrolledText(self, height=10, state="disabled",
                                              font=("Consolas", 9), wrap="word")
        self._log.pack(fill="x", padx=10, pady=(0, 10))

        # Seed with one row
        self._add_row()

    # ── Row Management ────────────────────────────────────────────────────────

    def _add_row(self):
        idx = len(self._rows) + 1
        row = VoicePairRow(self._pairs_container, idx, self._remove_row)
        self._rows.append(row)

    def _remove_row(self, row: VoicePairRow):
        if len(self._rows) == 1:
            messagebox.showinfo("VoiceClone", "At least one voice pair is required.")
            return
        self._rows.remove(row)
        row.destroy()
        # Renumber
        for i, r in enumerate(self._rows, 1):
            r.frame.winfo_children()[0].config(text=f"#{i}")

    # ── Settings Helpers ──────────────────────────────────────────────────────

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self._output_var.set(path)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_msg(self, text):
        self._log.config(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def _poll_log(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._log_msg(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate(self):
        # Validate rows first before doing anything
        for i, row in enumerate(self._rows, 1):
            err = row.validate()
            if err:
                messagebox.showerror("Validation error", f"Pair #{i}: {err}")
                return

        output_dir = self._output_var.get().strip()
        if not output_dir:
            messagebox.showerror("Validation error", "Output folder is required.")
            return

        self._gen_btn.config(state="disabled")

        def _run():
            tmp_files = []
            try:
                # Load model on first Generate press
                if not self._model_ready:
                    sys.stdout = StdoutRedirector(self._log_queue)
                    self._log_queue.put("Loading TTS model... (first run downloads ~2GB)")
                    from voice_clone_translator import VoiceCloneTranslator
                    self._cloner = VoiceCloneTranslator()
                    self._cloner.setup_tts_only()
                    self._model_ready = True
                    self._log_queue.put("Model ready. Starting generation...")

                os.makedirs(output_dir, exist_ok=True)
                self._log_queue.put(f"Generating → {output_dir}")
                pairs = []
                for row in self._rows:
                    text = row.get_text()
                    # Write inline text to a temp .txt file
                    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                                     delete=False, encoding="utf-8")
                    tmp.write(text)
                    tmp.close()
                    tmp_files.append(tmp.name)
                    pairs.append((tmp.name, row.voice_path, row.lang_var.get()))

                combined_path = os.path.join(output_dir, "combined.wav")
                results = self._cloner.batch_synthesize_multi_voice(
                    text_voice_pairs=pairs,
                    output_dir=output_dir,
                    language="en",
                    combined_output=combined_path,
                    add_silence=self._silence_var.get(),
                    speed=self._speed_var.get(),
                    temperature=self._temperature_var.get(),
                    top_p=self._top_p_var.get(),
                    repetition_penalty=self._rep_pen_var.get(),
                )
                duration = results.get("total_duration", 0)
                combined = results.get("combined_file", combined_path)
                self._log_queue.put(f"Done! Combined: {combined} ({duration:.1f}s)")
                self.after(0, lambda: messagebox.showinfo(
                    "Complete",
                    f"Generated successfully!\n\nCombined: {combined}\nDuration: {duration:.1f}s"
                ))
            except Exception as e:
                self._model_ready = False  # allow retry if model load failed
                self._cloner = None
                msg = str(e)
                self._log_queue.put(f"ERROR: {msg}")
                self.after(0, lambda m=msg: messagebox.showerror("Error", m))
            finally:
                for f in tmp_files:
                    try:
                        os.unlink(f)
                    except OSError:
                        pass
                self.after(0, lambda: self._gen_btn.config(state="normal"))

        threading.Thread(target=_run, daemon=True).start()


if __name__ == "__main__":
    app = VoiceCloneApp()
    app.mainloop()
