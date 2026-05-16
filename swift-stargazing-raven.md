# Plan: VoiceClone GUI .exe

## Context
User wants to convert the VoiceClone Python project into a standalone Windows .exe. The app uses Coqui TTS (XTTS-v2) + Faster-Whisper for voice cloning. The .exe needs a GUI where users can add multiple (text, reference voice) pairs, then generate a combined audio file. Models download on first run (~2GB), keeping the .exe small.

---

## Files to Create

1. `gui_app.py` — Main Tkinter GUI app
2. `voiceclone.spec` — PyInstaller build spec
3. `build.bat` — One-click build script
4. `download_ffmpeg.py` — Helper to fetch ffmpeg.exe static binary (run once before build)

## Files to Modify
- `voice_clone_translator.py` — add `setup_tts_only()` method (lines 39–60 area)

---

## Implementation Plan

### 1. `gui_app.py` — Tkinter GUI

**Structure:**
```
VoiceCloneApp (Tk root)
├── Header label
├── ScrollableFrame — holds VoicePairRow widgets
│   └── VoicePairRow × N
│       ├── Label "#N"
│       ├── Text input (multiline Entry/Text widget, ~3 lines)
│       ├── [Load .txt] button → filedialog.askopenfilename
│       ├── [Browse Voice] button → filedialog.askopenfilename (.wav/.mp3/.mp4/etc)
│       ├── Voice filename label (shows selected file)
│       ├── Language dropdown (OptionMenu — en/es/fr/de/it/pt/pl/tr/ru/nl/cs/ar/zh-cn/ja/hu/ko)
│       └── [✕] remove button
├── [+ Add Voice Pair] button
├── Separator
├── Settings row:
│   ├── Output dir: [Entry] [Browse]
│   └── Silence (s): [Spinbox 0.0–5.0]
├── [Generate] button
└── Log Text widget (scrollable, readonly) — progress/status output
```

**Key behaviors:**
- App startup: spawn background thread → `cloner.setup_tts_only()` → log "Models ready"
- First run with no cached models: TTS library auto-downloads to `~/.local/share/tts` — log stream via redirected stdout
- [Generate] button: disabled during model load and during generation
- Generation runs in background thread, uses queue to push log lines to GUI (thread-safe)
- Each VoicePairRow validates: text not empty AND voice file selected before generation starts
- After generation: show messagebox with combined output path + total duration

**Text input:** Small Text widget (3 rows). [Load .txt] populates it from file. User can also type directly.

**Voice file filter:** `[("Audio/Video files", "*.wav *.mp3 *.mp4 *.avi *.mov *.mkv *.flac *.ogg *.m4a")]`

**Generation call:**
```python
# Build pairs list from rows
pairs = [(row.get_text(), row.voice_path, row.language) for row in self.rows]
# Write temp .txt files for rows that have inline text (not file-backed)
# Call:
results = cloner.batch_synthesize_multi_voice(
    text_voice_pairs=pairs,
    output_dir=output_dir,
    language="en",
    combined_output=os.path.join(output_dir, "combined.wav"),
    add_silence=silence_val
)
```

For rows with inline text: write text to a temp `.txt` file, pass that path to the pair tuple, clean up after.

### 1b. `setup_tts_only()` — addition to `voice_clone_translator.py`

Add after existing `setup_models()` (line ~61):
```python
def setup_tts_only(self):
    """Initialize TTS model only — no Whisper or translator."""
    print("Loading TTS model...")
    from TTS.api import TTS
    self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
    print("TTS model loaded.")
```

GUI uses this. Keeps `setup_models()` untouched for CLI users.

---

### 2. `voiceclone.spec` — PyInstaller spec

Key settings:
- `Analysis`: add hidden imports for torch, TTS, faster_whisper, deep_translator, librosa, soundfile, pydub, moviepy, cutlet
- `binaries`: include `ffmpeg.exe` from project root → `('ffmpeg.exe', '.')`
- `datas`: include any TTS config files if needed
- `EXE`: `console=True` (keep console window for model download progress on first run), `name='VoiceClone'`
- Mode: `--onedir` (not onefile — faster startup, avoids large single-file extraction overhead)
- Entry point: `gui_app.py`

Critical hidden imports (Whisper + deep_translator excluded — GUI uses TTS only):
```python
hiddenimports=[
    'TTS', 'TTS.api', 'TTS.tts.configs', 'TTS.tts.models',
    'librosa', 'soundfile', 'pydub', 'pydub.audio_segment',
    'moviepy', 'moviepy.editor',
    'torch', 'torchaudio',
    'scipy', 'scipy.signal', 'scipy.io', 'scipy.io.wavfile',
    'cutlet', 'fugashi', 'unidic_lite',
    'sklearn', 'sklearn.utils._typedefs',
    'numba',
]
```

### 3. `build.bat`
```bat
@echo off
echo Building VoiceClone.exe...
python download_ffmpeg.py
pyinstaller voiceclone.spec --clean
echo Done. Output in dist\VoiceClone\
pause
```

### 4. `download_ffmpeg.py`
- Check if `ffmpeg.exe` exists in project root
- If not: download from https://github.com/BtbN/FFmpeg-Builds/releases static build
- Extract `ffmpeg.exe` only into project root

---

## FFmpeg Strategy
Bundle `ffmpeg.exe` (Windows static build) inside the dist folder. In `gui_app.py`, on startup, set:
```python
import os, sys
if getattr(sys, 'frozen', False):  # running as .exe
    ffmpeg_path = os.path.join(sys._MEIPASS, 'ffmpeg.exe')
    os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ['PATH']
```
This ensures pydub/moviepy/ffmpeg-python find the bundled binary.

---

## Verification
1. Run `python gui_app.py` directly first — verify GUI loads, model download works, generation produces audio
2. Run `build.bat` to produce `dist\VoiceClone\VoiceClone.exe`
3. Copy `dist\VoiceClone\` to a machine without Python installed, run `VoiceClone.exe`
4. First run: models download to `%USERPROFILE%\.local\share\tts` — verify log shows progress
5. Add 2 voice pairs (different voices, different text), click Generate, verify `combined.wav` plays correctly
