"""
Example: Multi-Voice Conversation
Generate a conversation with multiple speakers, each with their own voice.

NEW: Now supports ANY audio/video format!
- MP3, MP4, AVI, MOV, MKV, FLAC, OGG, WAV - all work!
- Files are automatically converted to WAV format internally

The language parameter MUST be specified to prevent auto-language-detection issues.
Always use the 3-tuple format: (text_file, reference_audio, language)
"""

from voice_clone_translator import VoiceCloneTranslator

# Initialize
cloner = VoiceCloneTranslator()
cloner.setup_models()

# Define text-voice pairs
# Format: [(text_file, reference_audio, language), ...]
# You can now use MP3, MP4, or any audio/video format - it auto-converts!
conversation = [
    ("speaker1_line1.txt", "test_audio.mp3", "en"),  # MP3 works!
    ("speaker2_line1.txt", "test_audio_marcus.mp3", "en"),  # Different voice
    ("speaker1_line2.txt", "test_audio.mp3", "en"),
    ("speaker2_line2.txt", "test_audio_marcus.mp3", "en"),  # Fixed typo
]

# Or mix formats:
# conversation = [
#     ("speaker1.txt", "voice1.mp3", "en"),  # MP3
#     ("speaker2.txt", "voice2.mp4", "en"),  # MP4
#     ("speaker3.txt", "voice3.wav", "en"),  # WAV
# ]

# Process with multi-voice synthesis
results = cloner.batch_synthesize_multi_voice(
    text_voice_pairs=conversation,
    output_dir="output/conversation",
    language="en",  # Default language (can be overridden per-pair)
    combined_output="output/conversation/full_conversation.wav",
    add_silence=0.5  # 0.5 seconds between speakers
)

print("\n✅ Conversation generated!")
print(f"\nIndividual clips:")
for clip in results['individual_files']:
    print(f"  - {clip}")

print(f"\n🎙️ Combined conversation: {results['combined_file']}")
print(f"⏱️ Total duration: {results['total_duration']:.2f} seconds")
