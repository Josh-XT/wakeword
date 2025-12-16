"""
Wake Word Model Architecture and Training

Implements a small, efficient neural network for wake word detection
that can be exported to various edge device formats.
"""
import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Audio processing constants
SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160  # 10ms at 16kHz
WIN_LENGTH = 400  # 25ms at 16kHz
MAX_AUDIO_LENGTH = 1.5  # seconds
MAX_FRAMES = int(MAX_AUDIO_LENGTH * SAMPLE_RATE / HOP_LENGTH)


@dataclass
class ModelConfig:
    """Configuration for wake word model."""
    word: str
    n_mfcc: int = N_MFCC
    sample_rate: int = SAMPLE_RATE
    max_length_sec: float = MAX_AUDIO_LENGTH
    model_type: str = "cnn"  # "cnn" or "gru"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ModelConfig":
        return cls(**data)


class AudioFeatureExtractor:
    """
    Extract MFCC features from audio for wake word detection.
    
    Designed to be reproducible across different platforms.
    """
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mfcc: int = N_MFCC,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # MFCC transform
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
                "n_mels": 80,
            }
        )
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC features from waveform.
        
        Args:
            waveform: Audio tensor of shape (1, samples) or (samples,)
            
        Returns:
            MFCC features of shape (n_mfcc, time_frames)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Ensure correct sample rate (resample if needed)
        mfcc = self.mfcc_transform(waveform)
        
        # Remove channel dimension if present
        if mfcc.dim() == 3:
            mfcc = mfcc.squeeze(0)
        
        return mfcc
    
    def load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load and preprocess audio file."""
        waveform, sr = torchaudio.load(str(audio_path))
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)
        
        return waveform
    
    def load_audio_bytes(self, audio_bytes: bytes) -> torch.Tensor:
        """Load audio from bytes."""
        import io
        buffer = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(buffer)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)
        
        return waveform


class WakeWordDataset(Dataset):
    """Dataset for wake word training."""
    
    def __init__(
        self,
        positive_samples: List[Tuple[bytes, Dict]],
        negative_samples: List[Tuple[bytes, Dict]],
        feature_extractor: AudioFeatureExtractor,
        max_frames: int = MAX_FRAMES,
    ):
        self.feature_extractor = feature_extractor
        self.max_frames = max_frames
        
        # Combine positive (label=1) and negative (label=0) samples
        self.samples = []
        
        for audio_bytes, metadata in positive_samples:
            self.samples.append((audio_bytes, 1, metadata))
        
        for audio_bytes, metadata in negative_samples:
            self.samples.append((audio_bytes, 0, metadata))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_bytes, label, metadata = self.samples[idx]
        
        # Load and extract features
        waveform = self.feature_extractor.load_audio_bytes(audio_bytes)
        features = self.feature_extractor.extract_features(waveform)
        
        # Pad or truncate to fixed length
        if features.shape[1] < self.max_frames:
            padding = torch.zeros(features.shape[0], self.max_frames - features.shape[1])
            features = torch.cat([features, padding], dim=1)
        else:
            features = features[:, :self.max_frames]
        
        return features, torch.tensor(label, dtype=torch.float32)


class WakeWordCNN(nn.Module):
    """
    Compact CNN for wake word detection.
    
    Designed to be small enough for edge deployment (~50-100KB).
    """
    
    def __init__(
        self,
        n_mfcc: int = N_MFCC,
        max_frames: int = MAX_FRAMES,
        hidden_size: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.n_mfcc = n_mfcc
        self.max_frames = max_frames
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Calculate flattened size
        h = n_mfcc // 8  # After 3 pooling layers
        w = max_frames // 8
        flat_size = 64 * h * w
        
        # Fully connected layers
        self.fc1 = nn.Linear(flat_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Convolutional blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x.squeeze(-1)


class WakeWordGRU(nn.Module):
    """
    Compact GRU for wake word detection.
    
    Alternative architecture, may work better for variable-length inputs.
    """
    
    def __init__(
        self,
        n_mfcc: int = N_MFCC,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.n_mfcc = n_mfcc
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(
            input_size=n_mfcc,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional
    
    def forward(self, x):
        # x shape: (batch, n_mfcc, time) -> (batch, time, n_mfcc)
        x = x.transpose(1, 2)
        
        # GRU forward
        output, _ = self.gru(x)
        
        # Use last hidden state
        x = output[:, -1, :]
        x = torch.sigmoid(self.fc(x))
        
        return x.squeeze(-1)


class NegativeSampleGenerator:
    """
    Generate negative samples for training.
    
    Creates samples of:
    - Background noise
    - Similar-sounding words
    - Random speech segments
    """
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
    
    def generate_noise_samples(self, count: int, duration: float = 1.0) -> List[Tuple[bytes, Dict]]:
        """Generate various types of noise samples."""
        samples = []
        
        for i in range(count):
            noise_type = np.random.choice(["white", "pink", "brown", "silence"])
            
            num_samples = int(duration * self.sample_rate)
            
            if noise_type == "white":
                audio = np.random.randn(num_samples) * 0.1
            elif noise_type == "pink":
                # Approximate pink noise
                white = np.random.randn(num_samples)
                b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
                a = [1, -2.494956002, 2.017265875, -0.522189400]
                from scipy.signal import lfilter
                audio = lfilter(b, a, white) * 0.3
            elif noise_type == "brown":
                # Brown noise (random walk)
                audio = np.cumsum(np.random.randn(num_samples)) * 0.001
                audio = audio / (np.abs(audio).max() + 1e-8) * 0.2
            else:  # silence with small noise
                audio = np.random.randn(num_samples) * 0.001
            
            # Convert to int16
            audio = (audio * 32767).astype(np.int16)
            
            # Convert to bytes
            import io
            import wave
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio.tobytes())
            
            buffer.seek(0)
            samples.append((
                buffer.read(),
                {"type": "noise", "noise_type": noise_type}
            ))
        
        return samples
    
    async def generate_similar_words(
        self, 
        target_word: str, 
        tts_generator,
        count: int = 50
    ) -> List[Tuple[bytes, Dict]]:
        """Generate samples of similar-sounding words."""
        # Common confusing words/sounds
        similar_patterns = [
            # Short words that might sound similar
            "hey", "hay", "hi", "he", "huh",
            "ok", "okay", "oh", "ow",
            "yeah", "yes", "yep", "no", "nope",
            # Common words
            "the", "a", "is", "it", "to", "and",
            "what", "that", "this", "there",
            # Numbers
            "one", "two", "three", "four", "five",
        ]
        
        # Add word fragments and variations
        if len(target_word) > 2:
            similar_patterns.extend([
                target_word[:-1],  # Missing last letter
                target_word[1:],   # Missing first letter
                target_word + "s", # Plural
                target_word + "ing",
            ])
        
        samples = []
        for word in similar_patterns[:count]:
            try:
                tts_samples = await tts_generator.generate_samples(word, target_count=2)
                for sample in tts_samples:
                    samples.append((
                        sample.audio_data,
                        {"type": "similar_word", "word": word}
                    ))
            except Exception as e:
                logger.debug(f"Could not generate sample for '{word}': {e}")
        
        return samples


class WakeWordTrainer:
    """
    Trainer for wake word models.
    
    Handles the full training pipeline including:
    - Dataset preparation
    - Model training
    - Validation
    - Export to various formats
    """
    
    def __init__(
        self,
        config: ModelConfig,
        device: Optional[str] = None,
    ):
        self.config = config
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=config.sample_rate,
            n_mfcc=config.n_mfcc,
        )
        
        # Create model
        if config.model_type == "gru":
            self.model = WakeWordGRU(
                n_mfcc=config.n_mfcc,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )
        else:
            self.model = WakeWordCNN(
                n_mfcc=config.n_mfcc,
                hidden_size=config.hidden_size,
                dropout=config.dropout,
            )
        
        self.model = self.model.to(self.device)
    
    def train(
        self,
        positive_samples: List[Tuple[bytes, Dict]],
        negative_samples: List[Tuple[bytes, Dict]],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Train the wake word model.
        
        Args:
            positive_samples: List of (audio_bytes, metadata) for positive class
            negative_samples: List of (audio_bytes, metadata) for negative class
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            validation_split: Fraction of data for validation
            progress_callback: Optional callback for progress updates
            
        Returns:
            Training history dict
        """
        # Create dataset
        dataset = WakeWordDataset(
            positive_samples,
            negative_samples,
            self.feature_extractor,
        )
        
        # Split into train/val
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        criterion = nn.BCELoss()
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
            
            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            
            if progress_callback:
                progress_callback(epoch + 1, epochs, {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                })
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        logger.info(f"Training complete. Best validation loss: {best_val_loss:.4f}")
        
        return history
    
    def evaluate(
        self,
        test_samples: List[Tuple[bytes, int]],
    ) -> Dict[str, float]:
        """Evaluate model on test samples."""
        self.model.eval()
        
        correct = 0
        total = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        with torch.no_grad():
            for audio_bytes, label in test_samples:
                waveform = self.feature_extractor.load_audio_bytes(audio_bytes)
                features = self.feature_extractor.extract_features(waveform)
                
                # Pad/truncate
                if features.shape[1] < MAX_FRAMES:
                    padding = torch.zeros(features.shape[0], MAX_FRAMES - features.shape[1])
                    features = torch.cat([features, padding], dim=1)
                else:
                    features = features[:, :MAX_FRAMES]
                
                features = features.unsqueeze(0).to(self.device)
                output = self.model(features)
                prediction = (output > 0.5).item()
                
                if prediction == label:
                    correct += 1
                    if label == 1:
                        true_positives += 1
                    else:
                        true_negatives += 1
                else:
                    if label == 1:
                        false_negatives += 1
                    else:
                        false_positives += 1
                
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
        }
    
    def predict(self, audio_bytes: bytes) -> Tuple[bool, float]:
        """
        Predict if audio contains the wake word.
        
        Returns:
            Tuple of (is_wake_word, confidence)
        """
        self.model.eval()
        
        with torch.no_grad():
            waveform = self.feature_extractor.load_audio_bytes(audio_bytes)
            features = self.feature_extractor.extract_features(waveform)
            
            # Pad/truncate
            if features.shape[1] < MAX_FRAMES:
                padding = torch.zeros(features.shape[0], MAX_FRAMES - features.shape[1])
                features = torch.cat([features, padding], dim=1)
            else:
                features = features[:, :MAX_FRAMES]
            
            features = features.unsqueeze(0).to(self.device)
            confidence = self.model(features).item()
            
            return confidence > 0.5, confidence
    
    def save(self, output_dir: Path) -> Dict[str, Path]:
        """
        Save model in multiple formats.
        
        Returns dict of format -> file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save PyTorch model
        torch_path = output_dir / "model.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
        }, torch_path)
        saved_files["pytorch"] = torch_path
        
        # Save config
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        saved_files["config"] = config_path
        
        # Export to ONNX
        try:
            onnx_path = output_dir / "model.onnx"
            dummy_input = torch.randn(1, self.config.n_mfcc, MAX_FRAMES).to(self.device)
            
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'},
                }
            )
            saved_files["onnx"] = onnx_path
            logger.info(f"Saved ONNX model to {onnx_path}")
        except Exception as e:
            logger.warning(f"Could not export ONNX: {e}")
        
        # Export to TFLite
        try:
            tflite_path = self._export_tflite(output_dir)
            if tflite_path:
                saved_files["tflite"] = tflite_path
        except Exception as e:
            logger.warning(f"Could not export TFLite: {e}")
        
        logger.info(f"Model saved to {output_dir}")
        return saved_files
    
    def _export_tflite(self, output_dir: Path) -> Optional[Path]:
        """Export model to TensorFlow Lite format."""
        try:
            import tensorflow as tf
            import onnx
            from onnx_tf.backend import prepare
            
            onnx_path = output_dir / "model.onnx"
            if not onnx_path.exists():
                return None
            
            # Load ONNX model
            onnx_model = onnx.load(str(onnx_path))
            tf_rep = prepare(onnx_model)
            
            # Save as SavedModel
            saved_model_dir = output_dir / "saved_model"
            tf_rep.export_graph(str(saved_model_dir))
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            tflite_path = output_dir / "model.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Saved TFLite model to {tflite_path}")
            return tflite_path
            
        except ImportError:
            logger.warning("TensorFlow not available for TFLite export")
            return None
        except Exception as e:
            logger.warning(f"TFLite export failed: {e}")
            return None
    
    @classmethod
    def load(cls, model_dir: Path, device: Optional[str] = None) -> "WakeWordTrainer":
        """Load a trained model."""
        model_dir = Path(model_dir)
        
        # Load config
        config_path = model_dir / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = ModelConfig.from_dict(config_dict)
        
        # Create trainer
        trainer = cls(config, device=device)
        
        # Load weights
        torch_path = model_dir / "model.pt"
        checkpoint = torch.load(torch_path, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        
        return trainer
