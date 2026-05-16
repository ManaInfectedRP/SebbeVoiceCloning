# VoiceClone Studio

Clone any voice and synthesize speech from text. Supports multiple voices, multi-language output, and a standalone Windows GUI.

## Features

- **Voice Cloning**: Coqui TTS (XTTS-v2) — zero-shot, no training required
- **Multi-Voice**: Generate conversations with multiple different voices
- **GUI**: Standalone Windows `.exe` with drag-and-drop workflow
- **CLI**: Full pipeline — transcribe, translate, and clone from video/audio files
- **17+ Languages**: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko
- **GPU Accelerated**: CUDA support for fast generation

## Requirements

- Python 3.11 (TTS does not support 3.12+)
- FFmpeg (see below)
- NVIDIA GPU recommended (CPU works but is slow)

---

## Installation

### 1. Install FFmpeg

**Windows:** Download from https://ffmpeg.org/download.html and add to PATH.

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

### 2. Create Python 3.11 virtual environment

```bash
py -3.11 -m venv venv311
```

### 3. Activate the venv

**Bash:**
```bash
source ./venv311/Scripts/activate
```

**PowerShell:**
```powershell
.\venv311\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads ~2GB of models automatically.

---

## GUI (Recommended)

### Run directly

```bash
py gui_app.py
```

### Build standalone .exe

```bash
build.bat
```

Output: `dist\VoiceClone\VoiceClone.exe` — share the whole `dist\VoiceClone\` folder.

### How to use the GUI

1. Type text (or click **Load .txt** to import a file) in a voice pair row
2. Click **Browse...** to select a reference voice file (any audio/video format)
3. Set the language per row
4. Click **+ Add Voice Pair** to add more speakers
5. Set output folder and silence gap between speakers
6. Click **Generate**

---

## CLI

### Single file (transcribe → translate → clone)

```bash
py voice_clone_translator.py input_video.mp4 --target-lang es
```

### Text-to-speech

```bash
py voice_clone_translator.py --text-mode my_text.txt \
    --reference-audio reference_voice.mp3 \
    --language en \
    --output my_speech.wav
```

### Multi-voice conversation

```bash
py multi_voice_example.py
```

### Batch processing

```bash
py voice_clone_translator.py --batch my_videos/ --target-lang fr --output-dir output
```

### Force GPU

```bash
py voice_clone_translator.py input.mp4 --target-lang de --device cuda
```

---

## Supported Languages

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `ru` | Russian |
| `es` | Spanish | `nl` | Dutch |
| `fr` | French | `cs` | Czech |
| `de` | German | `ar` | Arabic |
| `it` | Italian | `zh-cn` | Chinese |
| `pt` | Portuguese | `ja` | Japanese |
| `pl` | Polish | `hu` | Hungarian |
| `tr` | Turkish | `ko` | Korean |

---

## Performance

- **GPU**: Recommended — 4–10x faster than CPU
- **CPU**: Works but slow
- **RAM**: 4GB minimum, 8GB+ recommended
- **First run**: Downloads XTTS-v2 model (~2GB)

---

## License

- Coqui TTS: Mozilla Public License 2.0
- Faster Whisper: MIT License
- deep-translator: Apache License 2.0
