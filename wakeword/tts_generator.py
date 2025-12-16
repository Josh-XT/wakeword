"""
TTS Sample Generator - Multi-engine TTS for generating diverse wake word samples.

Uses multiple TTS engines in parallel to generate varied training samples:
- gTTS (Google Text-to-Speech) - Fast, good quality
- edge-tts (Microsoft Edge) - High quality, many voices
- pyttsx3 - Offline, system voices
- Chatterbox (optional) - Voice cloning capable
"""

import os
import io
import asyncio
import hashlib
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSEngine(Enum):
    GTTS = "gtts"
    EDGE = "edge"
    PYTTSX3 = "pyttsx3"
    CHATTERBOX = "chatterbox"


@dataclass
class TTSSample:
    """Represents a generated TTS sample."""

    audio_data: bytes
    sample_rate: int
    engine: TTSEngine
    voice: str
    word: str
    variation: str  # e.g., "normal", "slow", "fast"


class SampleGenerator:
    """
    Multi-engine TTS sample generator for wake word training.

    Generates diverse audio samples using multiple TTS engines and voices,
    then applies augmentation for additional variety.
    """

    # Voice configurations for different engines
    EDGE_VOICES = [
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-US-AriaNeural",
        "en-US-DavisNeural",
        "en-US-AmberNeural",
        "en-US-AnaNeural",
        "en-US-AndrewNeural",
        "en-US-BrandonNeural",
        "en-US-ChristopherNeural",
        "en-US-CoraNeural",
        "en-US-ElizabethNeural",
        "en-US-EricNeural",
        "en-US-JacobNeural",
        "en-US-MichelleNeural",
        "en-US-MonicaNeural",
        "en-US-RogerNeural",
        "en-US-SaraNeural",
        "en-US-SteffanNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
        "en-GB-LibbyNeural",
        "en-GB-MaisieNeural",
        "en-AU-NatashaNeural",
        "en-AU-WilliamNeural",
        "en-IN-NeerjaNeural",
        "en-IN-PrabhatNeural",
    ]

    GTTS_LANGUAGES = ["en", "en-au", "en-uk", "en-us", "en-ca", "en-in", "en-ie", "en-za"]

    def __init__(
        self,
        output_dir: Path,
        enable_gtts: bool = True,
        enable_edge: bool = True,
        enable_pyttsx3: bool = True,
        enable_chatterbox: bool = False,
        chatterbox_voices_dir: Optional[Path] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_gtts = enable_gtts
        self.enable_edge = enable_edge
        self.enable_pyttsx3 = enable_pyttsx3
        self.enable_chatterbox = enable_chatterbox
        self.chatterbox_voices_dir = chatterbox_voices_dir

        # Thread pool for parallel generation
        self.executor = ThreadPoolExecutor(max_workers=8)

        # Initialize engines
        self._init_engines()

    def _init_engines(self):
        """Initialize available TTS engines."""
        self.available_engines = []

        if self.enable_gtts:
            try:
                from gtts import gTTS

                self.available_engines.append(TTSEngine.GTTS)
                logger.info("gTTS engine initialized")
            except ImportError:
                logger.warning("gTTS not available")

        if self.enable_edge:
            try:
                import edge_tts

                self.available_engines.append(TTSEngine.EDGE)
                logger.info("Edge TTS engine initialized")
            except ImportError:
                logger.warning("edge-tts not available")

        if self.enable_pyttsx3:
            try:
                import pyttsx3

                self.pyttsx3_engine = pyttsx3.init()
                self.available_engines.append(TTSEngine.PYTTSX3)
                logger.info("pyttsx3 engine initialized")
            except Exception as e:
                logger.warning(f"pyttsx3 not available: {e}")

        if self.enable_chatterbox and self.chatterbox_voices_dir:
            try:
                from chatterbox.tts import ChatterboxTTS

                self.chatterbox_model = ChatterboxTTS.from_pretrained(device="cuda")
                self.available_engines.append(TTSEngine.CHATTERBOX)
                logger.info("Chatterbox TTS engine initialized")
            except Exception as e:
                logger.warning(f"Chatterbox not available: {e}")

    async def generate_gtts_sample(self, word: str, lang: str = "en") -> Optional[TTSSample]:
        """Generate a sample using Google TTS."""
        try:
            from gtts import gTTS

            tts = gTTS(text=word, lang=lang, slow=False)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)

            # Convert MP3 to WAV
            audio = AudioSegment.from_mp3(buffer)
            audio = audio.set_frame_rate(16000).set_channels(1)

            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            return TTSSample(
                audio_data=wav_buffer.read(),
                sample_rate=16000,
                engine=TTSEngine.GTTS,
                voice=lang,
                word=word,
                variation="normal",
            )
        except Exception as e:
            logger.error(f"gTTS generation failed: {e}")
            return None

    async def generate_edge_sample(self, word: str, voice: str) -> Optional[TTSSample]:
        """Generate a sample using Microsoft Edge TTS."""
        try:
            import edge_tts

            communicate = edge_tts.Communicate(word, voice)
            buffer = io.BytesIO()

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buffer.write(chunk["data"])

            buffer.seek(0)

            # Convert to WAV at 16kHz
            audio = AudioSegment.from_mp3(buffer)
            audio = audio.set_frame_rate(16000).set_channels(1)

            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            return TTSSample(
                audio_data=wav_buffer.read(),
                sample_rate=16000,
                engine=TTSEngine.EDGE,
                voice=voice,
                word=word,
                variation="normal",
            )
        except Exception as e:
            logger.error(f"Edge TTS generation failed for voice {voice}: {e}")
            return None

    def generate_pyttsx3_sample(
        self, word: str, voice_id: Optional[str] = None
    ) -> Optional[TTSSample]:
        """Generate a sample using pyttsx3 (offline)."""
        try:
            import pyttsx3

            engine = pyttsx3.init()

            if voice_id:
                engine.setProperty("voice", voice_id)

            # Save to temp file
            temp_path = self.output_dir / f"temp_{uuid.uuid4().hex}.wav"
            engine.save_to_file(word, str(temp_path))
            engine.runAndWait()

            # Load and convert
            audio = AudioSegment.from_wav(str(temp_path))
            audio = audio.set_frame_rate(16000).set_channels(1)

            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            # Cleanup
            temp_path.unlink(missing_ok=True)

            return TTSSample(
                audio_data=wav_buffer.read(),
                sample_rate=16000,
                engine=TTSEngine.PYTTSX3,
                voice=voice_id or "default",
                word=word,
                variation="normal",
            )
        except Exception as e:
            logger.error(f"pyttsx3 generation failed: {e}")
            return None

    async def generate_chatterbox_sample(self, word: str, voice_file: Path) -> Optional[TTSSample]:
        """Generate a sample using Chatterbox TTS with voice cloning."""
        try:
            if not hasattr(self, "chatterbox_model"):
                return None

            wav = self.chatterbox_model.generate(word, audio_prompt_path=str(voice_file))

            # Convert tensor to bytes
            import torch
            import torchaudio

            temp_path = self.output_dir / f"temp_cb_{uuid.uuid4().hex}.wav"

            if isinstance(wav, torch.Tensor):
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                torchaudio.save(str(temp_path), wav.cpu(), self.chatterbox_model.sr)

            # Load and resample to 16kHz
            audio = AudioSegment.from_wav(str(temp_path))
            audio = audio.set_frame_rate(16000).set_channels(1)

            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            temp_path.unlink(missing_ok=True)

            return TTSSample(
                audio_data=wav_buffer.read(),
                sample_rate=16000,
                engine=TTSEngine.CHATTERBOX,
                voice=voice_file.stem,
                word=word,
                variation="cloned",
            )
        except Exception as e:
            logger.error(f"Chatterbox generation failed: {e}")
            return None

    async def generate_samples(
        self, word: str, target_count: int = 500, progress_callback: Optional[callable] = None
    ) -> List[TTSSample]:
        """
        Generate diverse TTS samples for a wake word.

        Args:
            word: The wake word to generate samples for
            target_count: Target number of base samples (before augmentation)
            progress_callback: Optional callback for progress updates

        Returns:
            List of generated TTSSample objects
        """
        samples = []
        tasks = []

        # Calculate samples per engine
        enabled_count = len(self.available_engines)
        if enabled_count == 0:
            raise RuntimeError("No TTS engines available")

        samples_per_engine = target_count // enabled_count

        # Generate from gTTS
        if TTSEngine.GTTS in self.available_engines:
            for lang in self.GTTS_LANGUAGES[:samples_per_engine]:
                tasks.append(self.generate_gtts_sample(word, lang))

        # Generate from Edge TTS (most voices)
        if TTSEngine.EDGE in self.available_engines:
            for voice in self.EDGE_VOICES[:samples_per_engine]:
                tasks.append(self.generate_edge_sample(word, voice))

        # Generate from pyttsx3
        if TTSEngine.PYTTSX3 in self.available_engines:
            try:
                import pyttsx3

                engine = pyttsx3.init()
                voices = engine.getProperty("voices")
                for voice in voices[:samples_per_engine]:
                    # Run in thread since pyttsx3 is synchronous
                    loop = asyncio.get_event_loop()
                    tasks.append(
                        loop.run_in_executor(
                            self.executor, self.generate_pyttsx3_sample, word, voice.id
                        )
                    )
            except Exception as e:
                logger.warning(f"Could not enumerate pyttsx3 voices: {e}")

        # Generate from Chatterbox
        if TTSEngine.CHATTERBOX in self.available_engines and self.chatterbox_voices_dir:
            voice_files = list(self.chatterbox_voices_dir.glob("*.wav"))
            for voice_file in voice_files[:samples_per_engine]:
                tasks.append(self.generate_chatterbox_sample(word, voice_file))

        # Execute all tasks
        logger.info(f"Generating {len(tasks)} base samples for '{word}'...")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, TTSSample):
                samples.append(result)
            elif isinstance(result, Exception):
                logger.debug(f"Sample generation failed: {result}")

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, len(tasks), "generating")

        logger.info(f"Generated {len(samples)} base samples")
        return samples

    def save_samples(self, samples: List[TTSSample], word: str) -> Path:
        """
        Save samples to disk organized by word.

        Returns the directory containing the samples.
        """
        word_dir = self.output_dir / word.lower().replace(" ", "_")
        word_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(samples):
            filename = f"{sample.engine.value}_{sample.voice}_{i:04d}.wav"
            filepath = word_dir / filename

            with open(filepath, "wb") as f:
                f.write(sample.audio_data)

        logger.info(f"Saved {len(samples)} samples to {word_dir}")
        return word_dir


class AudioAugmenter:
    """
    Audio augmentation for expanding training dataset variety.

    Applies various transformations to base samples to create more
    training data with realistic variations.
    """

    def __init__(self):
        pass

    def augment_sample(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> List[Tuple[bytes, str]]:
        """
        Apply multiple augmentations to a single sample.

        Returns list of (augmented_audio_bytes, augmentation_name) tuples.
        """
        augmented = []

        # Load audio
        audio = AudioSegment.from_wav(io.BytesIO(audio_data))

        # Speed variations
        for speed in [0.85, 0.9, 1.1, 1.15]:
            aug = self._change_speed(audio, speed)
            augmented.append((self._to_bytes(aug), f"speed_{speed}"))

        # Pitch variations
        for semitones in [-2, -1, 1, 2]:
            aug = self._change_pitch(audio, semitones)
            augmented.append((self._to_bytes(aug), f"pitch_{semitones}"))

        # Volume variations
        for db in [-6, -3, 3, 6]:
            aug = audio + db
            augmented.append((self._to_bytes(aug), f"volume_{db}db"))

        # Add background noise
        for noise_level in [0.005, 0.01, 0.02]:
            aug = self._add_noise(audio, noise_level)
            augmented.append((self._to_bytes(aug), f"noise_{noise_level}"))

        # Reverb simulation (simple echo)
        aug = self._add_reverb(audio)
        augmented.append((self._to_bytes(aug), "reverb"))

        return augmented

    def _change_speed(self, audio: AudioSegment, speed: float) -> AudioSegment:
        """Change playback speed without changing pitch (approximately)."""
        # Simple speed change - actual pitch-preserving would need more complex processing
        new_sample_rate = int(audio.frame_rate * speed)
        return audio._spawn(
            audio.raw_data, overrides={"frame_rate": new_sample_rate}
        ).set_frame_rate(audio.frame_rate)

    def _change_pitch(self, audio: AudioSegment, semitones: int) -> AudioSegment:
        """Shift pitch by semitones."""
        new_sample_rate = int(audio.frame_rate * (2 ** (semitones / 12)))
        pitched = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
        return pitched.set_frame_rate(audio.frame_rate)

    def _add_noise(self, audio: AudioSegment, noise_level: float) -> AudioSegment:
        """Add white noise to audio."""
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        noise = np.random.normal(0, noise_level * 32767, len(samples))
        noisy = np.clip(samples + noise, -32768, 32767).astype(np.int16)

        return audio._spawn(noisy.tobytes())

    def _add_reverb(self, audio: AudioSegment, decay: float = 0.3) -> AudioSegment:
        """Add simple reverb effect."""
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        delay_samples = int(0.03 * audio.frame_rate)  # 30ms delay

        reverb = np.zeros(len(samples) + delay_samples)
        reverb[: len(samples)] = samples
        reverb[delay_samples:] += samples * decay

        reverb = np.clip(reverb[: len(samples)], -32768, 32767).astype(np.int16)
        return audio._spawn(reverb.tobytes())

    def _to_bytes(self, audio: AudioSegment) -> bytes:
        """Convert AudioSegment to WAV bytes."""
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        return buffer.read()

    async def augment_dataset(
        self,
        samples: List[TTSSample],
        augmentations_per_sample: int = 5,
        progress_callback: Optional[callable] = None,
    ) -> List[Tuple[bytes, Dict]]:
        """
        Augment an entire dataset of samples.

        Args:
            samples: Base TTS samples to augment
            augmentations_per_sample: How many augmented versions per base sample
            progress_callback: Optional progress callback

        Returns:
            List of (audio_bytes, metadata_dict) tuples
        """
        augmented_dataset = []

        # Include original samples
        for sample in samples:
            augmented_dataset.append(
                (
                    sample.audio_data,
                    {
                        "engine": sample.engine.value,
                        "voice": sample.voice,
                        "word": sample.word,
                        "augmentation": "original",
                    },
                )
            )

        total = len(samples)
        for i, sample in enumerate(samples):
            augmentations = self.augment_sample(sample.audio_data, sample.sample_rate)

            # Take a subset of augmentations
            selected = augmentations[:augmentations_per_sample]

            for aug_data, aug_name in selected:
                augmented_dataset.append(
                    (
                        aug_data,
                        {
                            "engine": sample.engine.value,
                            "voice": sample.voice,
                            "word": sample.word,
                            "augmentation": aug_name,
                        },
                    )
                )

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, total, "augmenting")

        logger.info(f"Created {len(augmented_dataset)} total samples after augmentation")
        return augmented_dataset
