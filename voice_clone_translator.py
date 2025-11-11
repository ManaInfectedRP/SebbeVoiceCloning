"""
Voice Cloning and Translation System

This program uses deep learning to:
1. Extract audio from MP4 files
2. Transcribe speech to text using Faster Whisper (4-5x faster than OpenAI Whisper)
3. Translate to target language
4. Clone the original voice and synthesize in the new language
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union
import warnings
warnings.filterwarnings('ignore')

class VoiceCloneTranslator:
    """
    Main class for voice cloning with translation capabilities.
    Uses Coqui TTS for voice cloning and Faster Whisper for speech recognition.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the voice cloning system.
        
        Args:
            device: 'cuda' or 'cpu'. Auto-detects if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tts_model = None
        self.whisper_model = None
        self.translator = None
        
    def setup_models(self):
        """Initialize all required models."""
        print("Loading models... This may take a few minutes on first run.")
        
        # Import here to avoid loading if not needed
        from TTS.api import TTS
        from deep_translator import GoogleTranslator
        from faster_whisper import WhisperModel
        
        # Load TTS model for voice cloning (XTTS-v2 is state-of-the-art)
        print("Loading TTS model for voice cloning...")
        self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        
        # Load Faster Whisper for speech recognition (4-5x faster than OpenAI Whisper)
        print("Loading Faster Whisper model for speech recognition...")
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.whisper_model = WhisperModel("base", device=self.device, compute_type=compute_type)
        
        # Setup translator
        self.translator = GoogleTranslator
        
        print("All models loaded successfully!")
        
    def extract_audio_from_mp4(self, video_path: str, output_audio_path: str) -> str:
        """
        Extract audio from MP4 video file.
        
        Args:
            video_path: Path to input video file
            output_audio_path: Path for output audio file
            
        Returns:
            Path to extracted audio file
        """
        from pydub import AudioSegment
        import moviepy.editor as mp
        
        print(f"Extracting audio from {video_path}...")
        
        try:
            # Extract audio using moviepy
            video = mp.VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(output_audio_path, codec='pcm_s16le')
            video.close()
            
            print(f"Audio extracted to {output_audio_path}")
            return output_audio_path
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> dict:
        """
        Transcribe audio to text using Faster Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription and detected language
        """
        print(f"Transcribing audio from {audio_path}...")
        
        # Transcribe with faster-whisper
        segments, info = self.whisper_model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True  # Voice activity detection for better accuracy
        )
        
        # Collect all segments
        transcription = " ".join([segment.text for segment in segments])
        language = info.language
        
        print(f"Detected language: {language}")
        print(f"Transcription: {transcription[:100]}...")
        
        return {
            'text': transcription,
            'language': language,
            'full_result': info
        }
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'en', 'es', 'fr')
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        print(f"Translating from {source_lang} to {target_lang}...")
        
        translator = self.translator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        
        print(f"Translation: {translated[:100]}...")
        return translated
    
    def clone_voice(
        self,
        text: str,
        speaker_audio_path: str,
        output_path: str,
        language: str = "en"
    ) -> str:
        """
        Clone a voice and generate speech with the cloned voice.
        
        Args:
            text: Text to synthesize
            speaker_audio_path: Path to reference audio for voice cloning
            output_path: Path for output synthesized audio
            language: Language code for synthesis
            
        Returns:
            Path to generated audio file
        """
        print(f"Cloning voice and synthesizing speech...")
        
        # Generate speech with cloned voice
        self.tts_model.tts_to_file(
            text=text,
            speaker_wav=speaker_audio_path,
            language=language,
            file_path=output_path
        )
        
        print(f"Voice cloned audio saved to {output_path}")
        return output_path
    
    def process_pipeline(
        self,
        input_path: str,
        target_language: str = "es",
        output_dir: str = "output",
        keep_intermediates: bool = True
    ) -> dict:
        """
        Complete pipeline: extract audio, transcribe, translate, and clone voice.
        
        Args:
            input_path: Path to input video/audio file
            target_language: Target language for translation (e.g., 'es', 'fr', 'de')
            output_dir: Directory for output files
            keep_intermediates: Whether to keep intermediate files
            
        Returns:
            Dictionary with paths to all generated files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        input_path = Path(input_path)
        base_name = input_path.stem
        
        # Step 1: Extract audio if input is video
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            audio_path = os.path.join(output_dir, f"{base_name}_extracted.wav")
            audio_path = self.extract_audio_from_mp4(str(input_path), audio_path)
        else:
            audio_path = str(input_path)
        
        # Step 2: Transcribe audio
        transcription = self.transcribe_audio(audio_path)
        
        # Save transcription
        transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(f"Original Language: {transcription['language']}\n")
            f.write(f"Transcription:\n{transcription['text']}\n")
        
        # Step 3: Translate text
        translated_text = self.translate_text(
            transcription['text'],
            source_lang=transcription['language'],
            target_lang=target_language
        )
        
        # Save translation
        translation_path = os.path.join(output_dir, f"{base_name}_translation.txt")
        with open(translation_path, 'w', encoding='utf-8') as f:
            f.write(f"Target Language: {target_language}\n")
            f.write(f"Translation:\n{translated_text}\n")
        
        # Step 4: Clone voice and synthesize in target language
        cloned_output_path = os.path.join(output_dir, f"{base_name}_cloned_{target_language}.wav")
        self.clone_voice(
            text=translated_text,
            speaker_audio_path=audio_path,
            output_path=cloned_output_path,
            language=target_language
        )
        
        results = {
            'original_audio': audio_path,
            'transcript_file': transcript_path,
            'translation_file': translation_path,
            'cloned_audio': cloned_output_path,
            'original_text': transcription['text'],
            'translated_text': translated_text,
            'source_language': transcription['language'],
            'target_language': target_language
        }
        
        print("\n" + "="*50)
        print("Processing Complete!")
        print("="*50)
        print(f"Original Language: {results['source_language']}")
        print(f"Target Language: {results['target_language']}")
        print(f"Cloned voice audio: {results['cloned_audio']}")
        print("="*50 + "\n")
        
        return results


def main():
    """Example usage of the VoiceCloneTranslator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Cloning with Translation")
    parser.add_argument("input", help="Input video/audio file path")
    parser.add_argument(
        "--target-lang",
        default="es",
        help="Target language code (e.g., es, fr, de, it, pt)"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use for processing"
    )
    
    args = parser.parse_args()
    
    # Initialize and run
    cloner = VoiceCloneTranslator(device=args.device)
    cloner.setup_models()
    
    results = cloner.process_pipeline(
        input_path=args.input,
        target_language=args.target_lang,
        output_dir=args.output_dir
    )
    
    print("\nGenerated files:")
    for key, value in results.items():
        if key.endswith('_file') or key.endswith('_audio'):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
