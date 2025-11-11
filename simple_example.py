"""
Simple example showing how to use the Voice Cloning Translator
"""

from voice_clone_translator import VoiceCloneTranslator

def simple_example():
    """
    Basic example: Clone a voice and translate to Spanish
    """
    # Initialize the system
    print("Initializing Voice Cloning System...")
    cloner = VoiceCloneTranslator()
    cloner.setup_models()
    
    # Example 1: Process a video file
    # Replace with your actual file path
    input_file = "your_video.mp4"  # or "your_audio.wav"
    
    # Process: extract audio, transcribe, translate to Spanish, clone voice
    results = cloner.process_pipeline(
        input_path=input_file,
        target_language="es",  # Spanish
        output_dir="output"
    )
    
    print("\n✓ Processing complete!")
    print(f"Original text: {results['original_text'][:100]}...")
    print(f"Translated text: {results['translated_text'][:100]}...")
    print(f"Cloned audio file: {results['cloned_audio']}")


def advanced_example():
    """
    Advanced example: Step-by-step control
    """
    cloner = VoiceCloneTranslator()
    cloner.setup_models()
    
    # Step 1: Extract audio from video
    audio_file = cloner.extract_audio_from_mp4(
        "input_video.mp4",
        "extracted_audio.wav"
    )
    
    # Step 2: Transcribe
    transcription = cloner.transcribe_audio(audio_file)
    print(f"Original: {transcription['text']}")
    
    # Step 3: Translate to multiple languages
    for lang_code, lang_name in [('es', 'Spanish'), ('fr', 'French'), ('de', 'German')]:
        translated = cloner.translate_text(
            transcription['text'],
            source_lang=transcription['language'],
            target_lang=lang_code
        )
        
        # Step 4: Clone voice in each language
        output_file = f"cloned_{lang_code}.wav"
        cloner.clone_voice(
            text=translated,
            speaker_audio_path=audio_file,
            output_path=output_file,
            language=lang_code
        )
        print(f"✓ Created {lang_name} version: {output_file}")


if __name__ == "__main__":
    print("Choose example:")
    print("1. Simple example (single translation)")
    print("2. Advanced example (multiple languages)")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        simple_example()
    elif choice == "2":
        advanced_example()
    else:
        print("Running simple example by default...")
        simple_example()
