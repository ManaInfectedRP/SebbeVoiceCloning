# Voice Cloning with Translation

A Python program that uses deep learning to clone voices and translate speech into different languages while maintaining the original speaker's voice characteristics.

## Features

- 🎤 **Voice Cloning**: Uses state-of-the-art Coqui TTS (XTTS-v2) for voice cloning
- 🎬 **Video Support**: Extract audio from MP4 and other video formats
- 🗣️ **Speech Recognition**: Automatic transcription using OpenAI Whisper
- 🌍 **Multi-language Translation**: Translate to 100+ languages using deep-translator
- 🔊 **Voice Synthesis**: Generate speech in target language with cloned voice

## Technology Stack

- **Voice Cloning**: Coqui TTS (XTTS-v2) - multilingual voice cloning
- **Speech Recognition**: Faster Whisper - automatic speech recognition (4-5x faster than OpenAI Whisper)
- **Translation**: deep-translator (Google Translate API)
- **Audio Processing**: librosa, pydub, moviepy
- **Deep Learning**: PyTorch

## Installation

### 1. Install FFmpeg (Required)

**Windows:**
- Download from https://ffmpeg.org/download.html
- Add to PATH

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note**: First run will download models (~2GB), which may take some time.

## Usage

### Command Line

Basic usage:
```bash
python voice_clone_translator.py input_video.mp4 --target-lang es
```

Advanced options:
```bash
python voice_clone_translator.py input_video.mp4 \
    --target-lang fr \
    --output-dir my_output \
    --device cuda
```

### Python API

```python
from voice_clone_translator import VoiceCloneTranslator

# Initialize
cloner = VoiceCloneTranslator()
cloner.setup_models()

# Process video/audio file
results = cloner.process_pipeline(
    input_path="video.mp4",
    target_language="es",  # Spanish
    output_dir="output"
)

print(f"Cloned audio: {results['cloned_audio']}")
```

### Supported Languages

Common language codes:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese

[See full list of supported languages](https://github.com/nidhaloff/deep-translator#supported-languages)

## Examples

### Example 1: Translate English Video to Spanish

```python
from voice_clone_translator import VoiceCloneTranslator

cloner = VoiceCloneTranslator()
cloner.setup_models()

results = cloner.process_pipeline(
    input_path="english_speech.mp4",
    target_language="es"
)
```

### Example 2: Multiple Languages

```python
cloner = VoiceCloneTranslator()
cloner.setup_models()

# Extract audio once
audio = cloner.extract_audio_from_mp4("video.mp4", "audio.wav")
transcription = cloner.transcribe_audio(audio)

# Generate in multiple languages
for lang in ['es', 'fr', 'de', 'it']:
    translated = cloner.translate_text(
        transcription['text'],
        source_lang=transcription['language'],
        target_lang=lang
    )
    
    cloner.clone_voice(
        text=translated,
        speaker_audio_path=audio,
        output_path=f"output_{lang}.wav",
        language=lang
    )
```

### Example 3: Audio File Input

```python
# Works with audio files too (not just videos)
cloner = VoiceCloneTranslator()
cloner.setup_models()

results = cloner.process_pipeline(
    input_path="podcast.wav",
    target_language="ja"  # Japanese
)
```

## How It Works

1. **Audio Extraction**: Extracts audio from video files (MP4, AVI, MOV, etc.)
2. **Transcription**: Uses Whisper to convert speech to text and detect language
3. **Translation**: Translates text to target language using deep-translator
4. **Voice Cloning**: Uses XTTS-v2 to synthesize translated text in the original speaker's voice

## Model Details

### Coqui TTS (XTTS-v2)
- State-of-the-art multilingual voice cloning
- Supports 17+ languages
- Can clone voice from just a few seconds of audio
- Zero-shot voice cloning (no training required)

### Faster Whisper
- 4-5x faster than OpenAI Whisper with same accuracy
- Lower memory usage (uses CTranslate2)
- Robust speech recognition with VAD (Voice Activity Detection)
- Automatic language detection
- Handles multiple accents and audio qualities

## Performance

- **GPU Recommended**: Processing is much faster with CUDA-enabled GPU
- **CPU Mode**: Works on CPU but slower (5-10x)
- **Memory**: ~4GB RAM minimum, 8GB+ recommended
- **First Run**: Downloads models (~2GB), subsequent runs are faster

## Troubleshooting

### FFmpeg Not Found
Install FFmpeg and ensure it's in your system PATH.

### CUDA Out of Memory
Use CPU mode: `--device cpu` or reduce batch size.

### Poor Voice Quality
- Use higher quality input audio (clear speech, minimal background noise)
- Provide longer reference audio (10+ seconds recommended)

### Translation Issues
- Check language code is correct
- Some languages may have better quality than others
- Try alternative translation services if needed

## Limitations

- Voice cloning quality depends on input audio quality
- Some languages have better TTS support than others
- Translation accuracy depends on deep-translator backend
- Processing time scales with audio length

## Advanced Configuration

You can customize the models and settings:

```python
# Use different Whisper model (tiny, base, small, medium, large-v2, large-v3)
from faster_whisper import WhisperModel
compute_type = "float16" if torch.cuda.is_available() else "int8"
cloner.whisper_model = WhisperModel("medium", device="cuda", compute_type=compute_type)

# Use different TTS model
from TTS.api import TTS
cloner.tts_model = TTS("your_model_name")
```

## License

This project uses various open-source models and libraries. Please review their individual licenses:
- Coqui TTS: Mozilla Public License 2.0
- Faster Whisper: MIT License
- deep-translator: Apache License 2.0

## Contributing

Contributions welcome! Areas for improvement:
- Additional TTS model support
- Better voice quality preprocessing
- Batch processing optimization
- GUI interface

## Credits

Built with:
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [deep-translator](https://github.com/nidhaloff/deep-translator)
# SebbeVoiceCloning
