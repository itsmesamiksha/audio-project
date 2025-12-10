"""
Audio Emotion Recognition - Inference Script
Use trained model to predict emotions from audio files
"""

import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor
import torch.nn as nn
from transformers import Wav2Vec2Model
import argparse
import os

# ============================================================================
# MODEL DEFINITION (same as training)
# ============================================================================

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_labels, pretrained_model_name="facebook/wav2vec2-base"):
        super(EmotionRecognitionModel, self).__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        for param in self.wav2vec2.encoder.layers[-2:].parameters():
            param.requires_grad = True
        
        hidden_size = self.wav2vec2.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )
    
    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)
        logits = self.classifier(pooled)
        return logits

# ============================================================================
# EMOTION PREDICTOR CLASS
# ============================================================================

class EmotionPredictor:
    def __init__(self, model_path, device='cpu'):
        """
        Initialize emotion predictor
        
        Args:
            model_path: Path to trained model (.pth file)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.sample_rate = 16000
        self.max_length = 5  # seconds
        
        # Emotion labels (RAVDESS)
        self.id_to_emotion = {
            0: 'angry',
            1: 'calm',
            2: 'disgust',
            3: 'fearful',
            4: 'happy',
            5: 'neutral',
            6: 'sad',
            7: 'surprised'
        }
        
        # Load processor
        print("Loading Wav2Vec2 processor...")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        
        # Load model
        print("Loading model...")
        self.model = EmotionRecognitionModel(num_labels=len(self.id_to_emotion))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Pad or truncate
            max_samples = self.sample_rate * self.max_length
            if len(audio) < max_samples:
                audio = np.pad(audio, (0, max_samples - len(audio)))
            else:
                audio = audio[:max_samples]
            
            return audio
        except Exception as e:
            raise Exception(f"Error loading audio: {e}")
    
    def predict(self, audio_path):
        """
        Predict emotion from audio file
        
        Args:
            audio_path: Path to .wav audio file
            
        Returns:
            dict with prediction results
        """
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Process audio
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_values)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_id].item()
        
        # Get all probabilities
        all_probs = {
            self.id_to_emotion[i]: float(probabilities[0][i])
            for i in range(len(self.id_to_emotion))
        }
        
        return {
            'emotion': self.id_to_emotion[predicted_id],
            'confidence': float(confidence),
            'all_probabilities': all_probs
        }
    
    def predict_batch(self, audio_paths):
        """Predict emotions for multiple audio files"""
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                result['file'] = audio_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
        return results

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Audio Emotion Recognition')
    parser.add_argument('--audio', type=str, required=True, 
                        help='Path to audio file (.wav)')
    parser.add_argument('--model', type=str, default='emotion_model.pth',
                        help='Path to trained model')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--batch', action='store_true',
                        help='Process multiple files (audio path should be directory)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = EmotionPredictor(args.model, device=args.device)
    
    if args.batch:
        # Batch processing
        if not os.path.isdir(args.audio):
            print("Error: --batch requires audio path to be a directory")
            return
        
        audio_files = [
            os.path.join(args.audio, f) 
            for f in os.listdir(args.audio) 
            if f.endswith('.wav')
        ]
        
        print(f"\nProcessing {len(audio_files)} audio files...")
        results = predictor.predict_batch(audio_files)
        
        print("\n" + "="*80)
        print("BATCH RESULTS")
        print("="*80)
        for result in results:
            print(f"\nFile: {result['file']}")
            print(f"Emotion: {result['emotion']} (confidence: {result['confidence']:.2%})")
    
    else:
        # Single file processing
        if not os.path.exists(args.audio):
            print(f"Error: Audio file not found: {args.audio}")
            return
        
        print(f"\nProcessing: {args.audio}")
        result = predictor.predict(args.audio)
        
        print("\n" + "="*80)
        print("PREDICTION RESULT")
        print("="*80)
        print(f"Predicted Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nAll Probabilities:")
        for emotion, prob in sorted(result['all_probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True):
            print(f"  {emotion:12s}: {prob:6.2%} {'█' * int(prob * 50)}")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def usage_examples():
    """Example code for using the predictor"""
    
    # Example 1: Single prediction
    predictor = EmotionPredictor('emotion_model.pth', device='cpu')
    result = predictor.predict('sample_audio.wav')
    print(f"Emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    # Example 2: Batch prediction
    audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
    results = predictor.predict_batch(audio_files)
    for result in results:
        print(f"{result['file']}: {result['emotion']} ({result['confidence']:.2%})")
    
    # Example 3: Get all probabilities
    result = predictor.predict('sample_audio.wav')
    for emotion, prob in result['all_probabilities'].items():
        print(f"{emotion}: {prob:.2%}")

if __name__ == "__main__":
    main()