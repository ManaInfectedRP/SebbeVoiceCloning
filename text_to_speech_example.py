"""
Simple example of using saved reference audio for text-to-speech voice cloning.

This script shows how to:
1. Use a previously saved reference audio file
2. Load text from a file
3. Generate speech with the cloned voice
"""

from voice_clone_translator import VoiceCloneTranslator

# Initialize the voice cloner
cloner = VoiceCloneTranslator()
cloner.setup_models()

# Path to your saved reference audio
reference_audio = "test_audio.mp3"

# Path to your text file
text_file = "my_text.txt"

# Output path for the generated audio
output_audio = "output/my_custom_speech.wav"

# Language for synthesis
# en - English
# es - Spanish
# fr - French
# de - German
# it - Italian
# pt - Portuguese
# pl - Polish
# tr - Turkish
# ru - Russian
# nl - Dutch
# cs - Czech
# ar - Arabic
# zh-cn - Chinese (Simplified)
# ja - Japanese
# hu - Hungarian
# ko - Korean
language = "nl"

# Generate speech
print("Generating speech with cloned voice...")
result = cloner.synthesize_from_text_file(
    text_file_path=text_file,
    reference_audio_path=reference_audio,
    output_audio_path=output_audio,
    language=language
)

print(f"✅ Done! Generated audio saved to: {result}")
