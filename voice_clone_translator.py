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
        import subprocess
        
        print(f"Extracting audio from {video_path}...")
        
        try:
            # Extract audio using ffmpeg directly (more robust than moviepy)
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '44100',  # 44.1kHz sample rate
                '-ac', '2',  # Stereo
                '-y',  # Overwrite output file
                output_audio_path
            ]
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"Failed to extract audio: {result.stderr}")
            
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
    
    def _split_text_by_length(self, text: str, max_length: int = 200, language: str = "en") -> list:
        """
        Split text into chunks that respect sentence boundaries.
        
        Args:
            text: Text to split
            max_length: Maximum character length per chunk
            language: Language code to determine sentence boundaries
            
        Returns:
            List of text chunks
        """
        # Define sentence delimiters based on language
        if language in ['ja', 'zh']:
            # Japanese and Chinese use different punctuation
            delimiters = ['。', '！', '？', '、', '\n']
        else:
            delimiters = ['.', '!', '?', '\n', ';']
        
        # Split by sentences first
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in delimiters:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Group sentences into chunks under max_length
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If single sentence is too long, split it
                if len(sentence) > max_length:
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 <= max_length:
                            temp_chunk += (" " if temp_chunk else "") + word
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            temp_chunk = word
                    current_chunk = temp_chunk
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def clone_voice(
        self,
        text: str,
        speaker_audio_path: str,
        output_path: str,
        language: str = "en"
    ) -> str:
        """
        Clone a voice and generate speech with the cloned voice.
        Handles long texts by splitting them into chunks.
        
        Args:
            text: Text to synthesize
            speaker_audio_path: Path to reference audio for voice cloning
            output_path: Path for output synthesized audio
            language: Language code for synthesis
            
        Returns:
            Path to generated audio file
        """
        print(f"Cloning voice and synthesizing speech...")
        
        # Character limits for different languages (conservative estimates)
        char_limits = {
            'ja': 150,  # Japanese
            'zh': 150,  # Chinese
            'ko': 150,  # Korean
            'ar': 150,  # Arabic
            'default': 250
        }
        
        max_chars = char_limits.get(language, char_limits['default'])
        
        try:
            # Check if text needs to be split
            if len(text) > max_chars:
                print(f"Text length ({len(text)} chars) exceeds limit ({max_chars}). Splitting into chunks...")
                
                # Split text into manageable chunks
                chunks = self._split_text_by_length(text, max_chars, language)
                print(f"Split into {len(chunks)} chunks")
                
                # Generate audio for each chunk
                import soundfile as sf
                chunk_audios = []
                
                for i, chunk in enumerate(chunks):
                    print(f"  Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
                    temp_output = output_path.replace('.wav', f'_chunk_{i}.wav')
                    
                    self.tts_model.tts_to_file(
                        text=chunk,
                        speaker_wav=speaker_audio_path,
                        language=language,
                        file_path=temp_output
                    )
                    
                    # Read the generated audio
                    audio, sr = sf.read(temp_output)
                    chunk_audios.append(audio)
                    
                    # Clean up temporary file
                    os.remove(temp_output)
                
                # Concatenate all chunks
                print("Combining audio chunks...")
                combined_audio = np.concatenate(chunk_audios)
                
                # Save combined audio
                sf.write(output_path, combined_audio, sr)
                print(f"Combined {len(chunks)} chunks into final audio")
                
            else:
                # Text is short enough, generate directly
                self.tts_model.tts_to_file(
                    text=text,
                    speaker_wav=speaker_audio_path,
                    language=language,
                    file_path=output_path
                )
                
        except RuntimeError as e:
            if "MeCab" in str(e) and language == "ja":
                print("\n⚠️  Warning: MeCab not found for Japanese processing.")
                print("Attempting to use alternative method...")
                
                try:
                    # Alternative: use English TTS as fallback
                    print("Note: Falling back to English TTS. For proper Japanese support, install MeCab.")
                    print("See: https://github.com/polm/fugashi#installation")
                    
                    self.tts_model.tts_to_file(
                        text=text,
                        speaker_wav=speaker_audio_path,
                        language="en",  # Fallback to English
                        file_path=output_path
                    )
                except Exception as fallback_error:
                    print(f"Fallback also failed: {fallback_error}")
                    raise
            else:
                raise
        
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
