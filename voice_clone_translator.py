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
    
    def validate_input_file(self, filepath: str, file_type: str = "audio/video") -> dict:
        """
        Validate input file before processing.
        
        Args:
            filepath: Path to file to validate
            file_type: Type of file expected ('audio/video', 'audio', 'text')
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'exists': False,
            'size_mb': 0,
            'extension': '',
            'warnings': [],
            'errors': []
        }
        
        filepath = Path(filepath)
        
        # Check if file exists
        if not filepath.exists():
            result['errors'].append(f"File not found: {filepath}")
            return result
        
        result['exists'] = True
        
        # Check if it's a file (not directory)
        if not filepath.is_file():
            result['errors'].append(f"Path is not a file: {filepath}")
            return result
        
        # Get file size
        size_bytes = filepath.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        result['size_mb'] = size_mb
        
        # Check file size
        if size_mb == 0:
            result['errors'].append("File is empty (0 bytes)")
            return result
        
        if size_mb > 500:
            result['warnings'].append(f"Large file ({size_mb:.1f} MB). Processing may take a while...")
        
        # Check file extension
        extension = filepath.suffix.lower()
        result['extension'] = extension
        
        # Validate extension based on file type
        if file_type == "audio/video":
            valid_extensions = ['.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv', '.flac', '.ogg', '.m4a', '.webm']
            if extension not in valid_extensions:
                result['warnings'].append(
                    f"Unusual extension '{extension}'. Supported: {', '.join(valid_extensions)}"
                )
        elif file_type == "audio":
            valid_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
            if extension not in valid_extensions:
                result['warnings'].append(
                    f"Unusual audio extension '{extension}'. Supported: {', '.join(valid_extensions)}"
                )
        elif file_type == "text":
            valid_extensions = ['.txt', '.md']
            if extension not in valid_extensions:
                result['errors'].append(
                    f"Invalid text file extension '{extension}'. Use .txt or .md"
                )
                return result
        
        # File is valid if no errors
        result['valid'] = len(result['errors']) == 0
        
        return result
    
    def print_validation_result(self, validation: dict, filepath: str):
        """Print validation results in a user-friendly format."""
        if not validation['exists']:
            print(f"\n❌ ERROR: File not found!")
            print(f"   Path: {filepath}")
            print(f"\n💡 Tip: Check if the file path is correct and the file exists.")
            return
        
        if validation['errors']:
            print(f"\n❌ ERROR: Invalid file!")
            for error in validation['errors']:
                print(f"   - {error}")
            return
        
        if validation['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"   - {warning}")
        
        print(f"✅ File validated: {Path(filepath).name} ({validation['size_mb']:.1f} MB)")
    
    def process_directory(
        self,
        input_dir: str,
        target_language: str = "es",
        output_dir: str = "output",
        file_extensions: list = None,
        recursive: bool = False,
        save_reference_audio: bool = True
    ) -> dict:
        """
        Process all audio/video files in a directory.
        
        Args:
            input_dir: Directory containing input files
            target_language: Target language for translation
            output_dir: Output directory for generated files
            file_extensions: List of extensions to process (default: common audio/video formats)
            recursive: Whether to search subdirectories
            save_reference_audio: Whether to save reference audio for each file
            
        Returns:
            Dictionary with processing results
        """
        print("\n" + "="*70)
        print("BATCH PROCESSING MODE")
        print("="*70)
        
        if file_extensions is None:
            file_extensions = ['.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv', '.flac', '.ogg', '.m4a']
        
        input_path = Path(input_dir)
        
        # Validate input directory
        if not input_path.exists():
            print(f"❌ ERROR: Directory not found: {input_dir}")
            return {'success': False, 'error': 'Directory not found'}
        
        if not input_path.is_dir():
            print(f"❌ ERROR: Path is not a directory: {input_dir}")
            return {'success': False, 'error': 'Not a directory'}
        
        # Find all matching files
        if recursive:
            files = [f for ext in file_extensions for f in input_path.rglob(f'*{ext}')]
        else:
            files = [f for ext in file_extensions for f in input_path.glob(f'*{ext}')]
        
        if not files:
            print(f"\n❌ No files found with extensions: {', '.join(file_extensions)}")
            print(f"   In directory: {input_dir}")
            return {'success': False, 'error': 'No matching files found'}
        
        print(f"\n📁 Found {len(files)} file(s) to process")
        print(f"🌍 Target language: {target_language}")
        print(f"📂 Output directory: {output_dir}")
        print("="*70 + "\n")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each file
        results = {
            'total': len(files),
            'successful': [],
            'failed': [],
            'skipped': []
        }
        
        for i, file_path in enumerate(files, 1):
            print(f"\n{'='*70}")
            print(f"Processing [{i}/{len(files)}]: {file_path.name}")
            print(f"{'='*70}")
            
            try:
                # Validate file
                validation = self.validate_input_file(str(file_path), "audio/video")
                
                if not validation['valid']:
                    print(f"❌ Skipping invalid file: {file_path.name}")
                    for error in validation['errors']:
                        print(f"   - {error}")
                    results['skipped'].append({
                        'file': str(file_path),
                        'reason': validation['errors']
                    })
                    continue
                
                self.print_validation_result(validation, str(file_path))
                
                # Create subdirectory for this file's outputs
                file_output_dir = os.path.join(output_dir, file_path.stem)
                os.makedirs(file_output_dir, exist_ok=True)
                
                # Process the file
                result = self.process_pipeline(
                    input_path=str(file_path),
                    target_language=target_language,
                    output_dir=file_output_dir,
                    save_reference_audio=save_reference_audio
                )
                
                results['successful'].append({
                    'file': str(file_path),
                    'output': result
                })
                
                print(f"\n✅ Successfully processed: {file_path.name}")
                
            except KeyboardInterrupt:
                print(f"\n\n⚠️  Batch processing interrupted by user!")
                print(f"   Processed {len(results['successful'])} of {len(files)} files")
                break
                
            except Exception as e:
                print(f"\n❌ Error processing {file_path.name}: {e}")
                results['failed'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
                continue
        
        # Print summary
        print("\n" + "="*70)
        print("BATCH PROCESSING SUMMARY")
        print("="*70)
        print(f"Total files: {results['total']}")
        print(f"✅ Successful: {len(results['successful'])}")
        print(f"❌ Failed: {len(results['failed'])}")
        print(f"⏭️  Skipped: {len(results['skipped'])}")
        
        if results['successful']:
            print(f"\n✅ Successfully processed files:")
            for item in results['successful']:
                print(f"   - {Path(item['file']).name}")
        
        if results['failed']:
            print(f"\n❌ Failed files:")
            for item in results['failed']:
                print(f"   - {Path(item['file']).name}: {item['error']}")
        
        if results['skipped']:
            print(f"\n⏭️  Skipped files:")
            for item in results['skipped']:
                print(f"   - {Path(item['file']).name}")
        
        print("="*70 + "\n")
        
        results['success'] = True
        return results
        
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
    
    def convert_to_wav(self, input_path: str, output_path: str = None) -> str:
        """
        Convert any audio/video file to WAV format for voice cloning.
        Supports MP3, MP4, AVI, MOV, MKV, FLAC, OGG, etc.
        
        Args:
            input_path: Path to input audio/video file
            output_path: Path for output WAV file (optional, auto-generated if None)
            
        Returns:
            Path to converted WAV file
        """
        import subprocess
        from pathlib import Path
        
        input_path = Path(input_path)
        
        # Check if already a WAV file
        if input_path.suffix.lower() == '.wav':
            print(f"File is already WAV format: {input_path}")
            return str(input_path)
        
        # Generate output path if not provided
        if output_path is None:
            output_path = str(input_path.with_suffix('.wav'))
        
        print(f"Converting {input_path.suffix} to WAV format...")
        
        try:
            # Convert using ffmpeg
            command = [
                'ffmpeg',
                '-i', str(input_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '22050',  # 22.05kHz (good for voice)
                '-ac', '1',  # Mono (better for voice cloning)
                '-y',  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"Failed to convert audio: {result.stderr}")
            
            print(f"✅ Converted to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error converting audio: {e}")
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
            en - English
            es - Spanish
            fr - French
            de - German
            it - Italian
            pt - Portuguese
            pl - Polish
            tr - Turkish
            ru - Russian
            nl - Dutch
            cs - Czech
            ar - Arabic
            zh-cn - Chinese (Simplified)
            ja - Japanese
            hu - Hungarian
            ko - Korean
            
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
        language: str = "en",
        clear_cache: bool = False
    ) -> str:
        """
        Clone a voice and generate speech with the cloned voice.
        Handles long texts by splitting them into chunks.
        
        Args:
            text: Text to synthesize
            speaker_audio_path: Path to reference audio for voice cloning
            output_path: Path for output synthesized audio
            language: Language code for synthesis
            clear_cache: Force clear model cache (helps with language consistency)
            
        Returns:
            Path to generated audio file
        """
        print(f"Cloning voice and synthesizing speech...")
        print(f"  Target language: {language}")
        
        # Clear CUDA cache if requested (helps prevent language mixing on reused references)
        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  🔄 Cleared GPU cache for consistent generation")
        
        # Validate language is supported
        if self.tts_model and hasattr(self.tts_model, 'languages'):
            if language not in self.tts_model.languages:
                print(f"  ⚠️ Warning: '{language}' might not be supported. Available: {self.tts_model.languages}")
                print(f"  ⚠️ Forcing language to: {language}")
        
        try:
            # Check if text needs chunking (XTTS has 400 token limit ≈ 250-300 chars)
            max_chars = 250  # Conservative limit to stay under 400 tokens
            
            if len(text) > max_chars:
                print(f"Text length ({len(text)} chars) exceeds safe limit ({max_chars}). Splitting into chunks...")
                
                # Split text into manageable chunks
                chunks = self._split_text_by_length(text, max_chars, language)
                print(f"Split into {len(chunks)} chunks")
                
                # Generate audio for each chunk
                import soundfile as sf
                chunk_audios = []
                
                for i, chunk in enumerate(chunks):
                    print(f"  Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars) in {language}...")
                    temp_output = output_path.replace('.wav', f'_chunk_{i}.wav')
                    
                    # Try using tts() for better control
                    try:
                        wav = self.tts_model.tts(
                            text=chunk,
                            speaker_wav=speaker_audio_path,
                            language=language,
                            split_sentences=False
                        )
                        sf.write(temp_output, wav, 24000)
                    except (TypeError, AttributeError):
                        # Fallback to tts_to_file
                        try:
                            self.tts_model.tts_to_file(
                                text=chunk,
                                speaker_wav=speaker_audio_path,
                                language=language,
                                file_path=temp_output,
                                split_sentences=False
                            )
                        except TypeError:
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
                print(f"  Generating speech in language: {language}")
                
                # Use tts() method for more control, then save manually
                try:
                    print(f"  Using advanced generation mode for better language control...")
                    # Generate audio directly (returns numpy array)
                    wav = self.tts_model.tts(
                        text=text,
                        speaker_wav=speaker_audio_path,
                        language=language,
                        split_sentences=False  # Prevent per-sentence language detection
                    )
                    
                    # Save to file
                    import soundfile as sf
                    sf.write(output_path, wav, 24000)  # XTTS uses 24kHz
                    
                except (TypeError, AttributeError):
                    # Fallback to tts_to_file if tts() doesn't work as expected
                    print(f"  Using fallback generation mode...")
                    try:
                        self.tts_model.tts_to_file(
                            text=text,
                            speaker_wav=speaker_audio_path,
                            language=language,
                            file_path=output_path,
                            split_sentences=False
                        )
                    except TypeError:
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
    
    def synthesize_from_text_file(
        self,
        text_file_path: str,
        reference_audio_path: str,
        output_audio_path: str,
        language: str = "en"
    ) -> str:
        """
        Generate speech from a text file using a reference audio for voice cloning.
        
        Args:
            text_file_path: Path to text file containing the text to synthesize
            reference_audio_path: Path to reference audio for voice cloning
            output_audio_path: Path for output synthesized audio
            language: Language code for synthesis
            en - English
            es - Spanish
            fr - French
            de - German
            it - Italian
            pt - Portuguese
            pl - Polish
            tr - Turkish
            ru - Russian
            nl - Dutch
            cs - Czech
            ar - Arabic
            zh-cn - Chinese (Simplified)
            ja - Japanese
            hu - Hungarian
            ko - Korean
            
        Returns:
            Path to generated audio file
        """
        # Validate text file
        text_validation = self.validate_input_file(text_file_path, "text")
        if not text_validation['valid']:
            self.print_validation_result(text_validation, text_file_path)
            raise ValueError(f"Invalid text file: {text_validation['errors']}")
        
        # Validate reference audio
        audio_validation = self.validate_input_file(reference_audio_path, "audio/video")
        if not audio_validation['valid']:
            self.print_validation_result(audio_validation, reference_audio_path)
            raise ValueError(f"Invalid reference audio: {audio_validation['errors']}")
        
        # Read text from file
        print(f"Reading text from {text_file_path}...")
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            raise ValueError("Text file is empty!")
        
        print(f"Text length: {len(text)} characters")
        
        # Auto-convert to WAV if needed
        if not reference_audio_path.lower().endswith('.wav'):
            print(f"Reference audio is {Path(reference_audio_path).suffix}, converting to WAV...")
            converted_path = os.path.join(
                os.path.dirname(output_audio_path) or "output",
                f"_temp_reference_{Path(reference_audio_path).stem}.wav"
            )
            reference_audio_path = self.convert_to_wav(reference_audio_path, converted_path)
        
        # Generate speech with cloned voice
        result = self.clone_voice(
            text=text,
            speaker_audio_path=reference_audio_path,
            output_path=output_audio_path,
            language=language
        )
        
        # Clean up temporary converted file if it was created
        if "_temp_reference_" in reference_audio_path and os.path.exists(reference_audio_path):
            try:
                os.remove(reference_audio_path)
                print(f"Cleaned up temporary file: {reference_audio_path}")
            except:
                pass
        
        return result
    
    def batch_synthesize_multi_voice(
        self,
        text_voice_pairs: list,
        output_dir: str = "output",
        language: str = "en",
        combined_output: str = None,
        add_silence: float = 0.5
    ) -> dict:
        """
        Generate speech from multiple text files using different voices and combine them.
        
        Args:
            text_voice_pairs: List of tuples [(text_file_1, reference_audio_1), (text_file_2, reference_audio_2), ...]
                            OR [(text_file_1, reference_audio_1, language_1), (text_file_2, reference_audio_2, language_2), ...]
            output_dir: Directory for output files
            language: Default language code for synthesis (used if not specified per-pair)
            combined_output: Path for combined audio file (default: output_dir/combined_voices.wav)
            add_silence: Seconds of silence to add between speakers (default: 0.5)
            
        Returns:
            Dictionary with paths to individual and combined audio files
        """
        import soundfile as sf
        
        print("\n" + "="*60)
        print("Multi-Voice Batch Processing")
        print("="*60)
        print(f"Processing {len(text_voice_pairs)} text-voice pairs")
        print(f"Default Language: {language}")
        print("="*60 + "\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Pre-convert all unique reference audio files to WAV (prevents re-conversion issues)
        print("Pre-converting reference audio files...")
        reference_audio_cache = {}  # Cache converted references
        unique_references = set()
        
        for pair in text_voice_pairs:
            reference_audio = pair[1]  # Second element is always reference audio
            unique_references.add(reference_audio)
        
        for ref_audio in unique_references:
            if not ref_audio.lower().endswith('.wav'):
                print(f"  Converting: {Path(ref_audio).name}")
                converted_path = os.path.join(output_dir, f"_cached_ref_{Path(ref_audio).stem}.wav")
                reference_audio_cache[ref_audio] = self.convert_to_wav(ref_audio, converted_path)
            else:
                reference_audio_cache[ref_audio] = ref_audio
        
        print(f"  Cached {len(reference_audio_cache)} unique reference files\n")
        
        individual_outputs = []
        audio_segments = []
        sample_rate = None
        reference_usage_count = {}  # Track how many times each reference is used
        
        for i, pair in enumerate(text_voice_pairs, 1):
            # Support both 2-tuple and 3-tuple formats
            if len(pair) == 2:
                text_file, reference_audio = pair
                segment_language = language  # Use default
            elif len(pair) == 3:
                text_file, reference_audio, segment_language = pair
            else:
                raise ValueError(f"Invalid pair format at index {i}. Expected (text, audio) or (text, audio, language)")
            
            print(f"\n[{i}/{len(text_voice_pairs)}] Processing: {text_file}")
            print(f"  Voice reference: {reference_audio}")
            print(f"  Language: {segment_language}")
            
            # Generate output filename
            text_file_name = Path(text_file).stem
            output_path = os.path.join(output_dir, f"voice_{i}_{text_file_name}.wav")
            
            # Use cached (pre-converted) reference audio
            cached_reference = reference_audio_cache.get(reference_audio, reference_audio)
            
            # Track reference usage
            reference_usage_count[reference_audio] = reference_usage_count.get(reference_audio, 0) + 1
            usage_count = reference_usage_count[reference_audio]
            
            print(f"  Using cached reference: {Path(cached_reference).name} (use #{usage_count})")
            
            # Clear cache on 2nd+ use of same reference (prevents language mixing)
            should_clear_cache = usage_count > 1
            if should_clear_cache:
                print(f"  🔄 Clearing GPU cache (reused reference)")
            
            # Read text directly and call clone_voice (bypass synthesize_from_text_file to avoid re-conversion)
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                raise ValueError(f"Text file is empty: {text_file}")
            
            # Generate speech directly with cached reference
            self.clone_voice(
                text=text,
                speaker_audio_path=cached_reference,  # Use pre-converted WAV
                output_path=output_path,
                language=segment_language,
                clear_cache=should_clear_cache  # Clear cache for reused references
            )
            
            individual_outputs.append(output_path)
            
            # Read the generated audio for combining
            audio_data, sr = sf.read(output_path)
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                # Resample if needed (shouldn't happen with same TTS model)
                print(f"  Warning: Sample rate mismatch ({sr} vs {sample_rate})")
            
            audio_segments.append(audio_data)
            
            # Add silence between speakers (except after last one)
            if i < len(text_voice_pairs) and add_silence > 0:
                silence_samples = int(sample_rate * add_silence)
                silence = np.zeros(silence_samples)
                audio_segments.append(silence)
                print(f"  Added {add_silence}s silence")
        
        # Combine all audio segments
        print("\n" + "-"*60)
        print("Combining all audio segments...")
        combined_audio = np.concatenate(audio_segments)
        
        # Save combined audio
        if combined_output is None:
            combined_output = os.path.join(output_dir, "combined_voices.wav")
        
        sf.write(combined_output, combined_audio, sample_rate)
        print(f"Combined audio saved to: {combined_output}")
        
        results = {
            'individual_files': individual_outputs,
            'combined_file': combined_output,
            'total_duration': len(combined_audio) / sample_rate,
            'sample_rate': sample_rate,
            'num_voices': len(text_voice_pairs)
        }
        
        # Clean up cached reference files
        print("\nCleaning up cached reference files...")
        for original_ref, cached_ref in reference_audio_cache.items():
            if "_cached_ref_" in cached_ref and os.path.exists(cached_ref):
                try:
                    os.remove(cached_ref)
                    print(f"  Removed: {Path(cached_ref).name}")
                except Exception as e:
                    print(f"  Warning: Could not remove {cached_ref}: {e}")
        
        print("\n" + "="*60)
        print("Multi-Voice Processing Complete!")
        print("="*60)
        print(f"Individual files: {len(individual_outputs)}")
        for idx, file in enumerate(individual_outputs, 1):
            print(f"  {idx}. {file}")
        print(f"\nCombined file: {combined_output}")
        print(f"Total duration: {results['total_duration']:.2f} seconds")
        print("="*60 + "\n")
        
        return results
    
    def process_pipeline(
        self,
        input_path: str,
        target_language: str = "es",
        output_dir: str = "output",
        keep_intermediates: bool = True,
        save_reference_audio: bool = True
    ) -> dict:
        """
        Complete pipeline: extract audio, transcribe, translate, and clone voice.
        
        Args:
            input_path: Path to input video/audio file
            target_language: Target language for translation
            en - English
            es - Spanish
            fr - French
            de - German
            it - Italian
            pt - Portuguese
            pl - Polish
            tr - Turkish
            ru - Russian
            nl - Dutch
            cs - Czech
            ar - Arabic
            zh-cn - Chinese (Simplified)
            ja - Japanese
            hu - Hungarian
            ko - Korean
            output_dir: Directory for output files
            keep_intermediates: Whether to keep intermediate files
            
        Returns:
            Dictionary with paths to all generated files
        """
        # Validate input file
        validation = self.validate_input_file(input_path, "audio/video")
        self.print_validation_result(validation, input_path)
        
        if not validation['valid']:
            raise ValueError(f"Invalid input file: {validation['errors']}")
        
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
        
        # Save reference audio for future use
        reference_audio_path = None
        if save_reference_audio:
            reference_audio_path = os.path.join(output_dir, f"{base_name}_reference_voice.wav")
            if audio_path != reference_audio_path:
                import shutil
                shutil.copy(audio_path, reference_audio_path)
                print(f"Reference audio saved to: {reference_audio_path}")
                print("You can use this audio file later for voice cloning with custom text!")
        
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
            'reference_audio': reference_audio_path,
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
    
    parser = argparse.ArgumentParser(
        description="Voice Cloning with Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: video -> transcribe -> translate -> clone
  python voice_clone_translator.py video.mp4 --target-lang es
  
  # Batch process all files in a directory
  python voice_clone_translator.py --batch my_videos/ --target-lang fr --output-dir batch_output
  
  # Batch process with specific extensions and recursive search
  python voice_clone_translator.py --batch my_media/ --target-lang de --recursive --extensions .mp4 .mp3
  
  # Generate speech from text file using saved reference audio
  python voice_clone_translator.py --text-mode my_text.txt --reference-audio output/speaker_reference_voice.wav --language en --output my_speech.wav
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--text-mode",
        action="store_true",
        help="Text-to-speech mode: Generate speech from a text file using reference audio"
    )
    parser.add_argument(
        "--multi-voice",
        action="store_true",
        help="Multi-voice mode: Process multiple text files with different voices"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: Process all audio/video files in a directory"
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List all supported languages and exit"
    )
    
    # Main input (video/audio for pipeline mode, text file for text mode)
    parser.add_argument(
        "input",
        nargs="?",
        help="Input video/audio file (pipeline mode) or text file (text mode)"
    )
    
    # Pipeline mode arguments
    parser.add_argument(
        "--target-lang",
        default="es",
        help="Target language code for translation (e.g., es, fr, de, it, pt, ja)"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for generated files (pipeline mode)"
    )
    
    # Text mode arguments
    parser.add_argument(
        "--reference-audio",
        help="Path to reference audio for voice cloning (text mode)"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code for synthesis (text mode, e.g., en, es, fr, de, ja)"
    )
    parser.add_argument(
        "--output",
        help="Output audio file path (text mode)"
    )
    
    # Multi-voice mode arguments
    parser.add_argument(
        "--pairs",
        nargs="+",
        help="Text-voice pairs in format: text1.txt,voice1.wav text2.txt,voice2.wav (multi-voice mode)"
    )
    
    # Batch mode arguments
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process subdirectories recursively (batch mode)"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        help="File extensions to process (batch mode, e.g., .mp3 .mp4 .wav)"
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=0.5,
        help="Seconds of silence between speakers in combined audio (default: 0.5)"
    )
    parser.add_argument(
        "--combined-output",
        help="Path for combined audio file (multi-voice mode)"
    )
    
    # Common arguments
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use for processing"
    )
    
    args = parser.parse_args()
    
    # Initialize
    cloner = VoiceCloneTranslator(device=args.device)
    cloner.setup_models()
    
    # List languages mode
    if args.list_languages:
        cloner.list_supported_languages()
        return
    
    if args.batch:
        # Batch processing mode
        if not args.input:
            parser.error("Directory path is required in batch mode")
        
        # Parse extensions if provided
        extensions = None
        if args.extensions:
            extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in args.extensions]
        
        results = cloner.process_directory(
            input_dir=args.input,
            target_language=args.target_lang,
            output_dir=args.output_dir,
            file_extensions=extensions,
            recursive=args.recursive
        )
        
        if not results['success']:
            exit(1)
    
    elif args.multi_voice:
        # Multi-voice mode
        if not args.pairs:
            parser.error("--pairs is required in multi-voice mode")
        
        # Parse pairs (format: text1.txt,voice1.wav text2.txt,voice2.wav)
        text_voice_pairs = []
        for pair in args.pairs:
            parts = pair.split(',')
            if len(parts) != 2:
                parser.error(f"Invalid pair format: {pair}. Use: text.txt,voice.wav")
            text_file, voice_file = parts
            text_voice_pairs.append((text_file.strip(), voice_file.strip()))
        
        results = cloner.batch_synthesize_multi_voice(
            text_voice_pairs=text_voice_pairs,
            output_dir=args.output_dir,
            language=args.language,
            combined_output=args.combined_output,
            add_silence=args.silence
        )
        
    elif args.text_mode:
        # Text-to-speech mode
        if not args.input:
            parser.error("Text file path is required in text mode")
        if not args.reference_audio:
            parser.error("--reference-audio is required in text mode")
        if not args.output:
            parser.error("--output is required in text mode")
        
        print("\n" + "="*50)
        print("Text-to-Speech Mode")
        print("="*50)
        
        output_path = cloner.synthesize_from_text_file(
            text_file_path=args.input,
            reference_audio_path=args.reference_audio,
            output_audio_path=args.output,
            language=args.language
        )
        
        print("\n" + "="*50)
        print("Text-to-Speech Complete!")
        print("="*50)
        print(f"Generated audio: {output_path}")
        print("="*50 + "\n")
        
    else:
        # Full pipeline mode
        if not args.input:
            parser.error("Input file path is required")
        
        results = cloner.process_pipeline(
            input_path=args.input,
            target_language=args.target_lang,
            output_dir=args.output_dir
        )
        
        print("\nGenerated files:")
        for key, value in results.items():
            if value and (key.endswith('_file') or key.endswith('_audio')):
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
