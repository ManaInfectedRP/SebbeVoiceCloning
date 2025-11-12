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

**For Japanese Support**: If you want to use Japanese (ja) as target language, see [JAPANESE_SETUP.md](JAPANESE_SETUP.md) for additional setup steps.

## Usage

### Command Line

Basic usage:
```bash
py voice_clone_translator.py input_video.mp4 --target-lang es
```

```bash
py text_to_speech_example.py ( needs to modify reference_audio, text_file, output_audio)
```

```bash
py mutli_voice_exaxmple.py ( needs to edit conversation [Array])
```

Advanced options:
```bash
py voice_clone_translator.py input_video.mp4 \
    --target-lang fr \
    --output-dir my_output \
    --device cuda
```


### Supported Languages

Common language codes:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `pl` - Polish
- `tr` - Turkish
- `ru` - Russian
- `nl` - Dutch
- `cs` - Czech
- `ar` - Arabic
- `zh-cn` - Chinese (Simplified)
- `ja` - Japanese
- `hu` - Hungarian
- `ko` - Korean

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

## License

This project uses various open-source models and libraries. Please review their individual licenses:
- Coqui TTS: Mozilla Public License 2.0
- Faster Whisper: MIT License
- deep-translator: Apache License 2.0

## Credits

Built with:
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [deep-translator](https://github.com/nidhaloff/deep-translator)
# SebbeVoiceCloning
