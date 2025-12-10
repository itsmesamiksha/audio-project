"""
Audio Emotion Recognition - Demo & Test Script
Quick demo to test the trained model with visualizations
"""

import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Wav2Vec2Processor
import torch.nn as nn
from transformers import Wav2Vec2Model
import librosa.display

# ============================================================================
# MODEL DEFINITION
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
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_waveform(audio, sr, title="Audio Waveform"):
    """Plot audio waveform"""
    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(audio, sr=sr, alpha=0.8)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

def plot_spectrogram(audio, sr, title="Mel Spectrogram"):
    """Plot mel spectrogram"""
    plt.figure(figsize=(14, 5))
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_emotion_probabilities(probabilities, title="Emotion Probabilities"):
    """Plot emotion probabilities as bar chart"""
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
    bars = plt.bar(emotions, probs, color=colors, alpha=0.8)
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_mfcc(audio, sr, title="MFCC Features"):
    """Plot MFCC features"""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()
    plt.show()

# ============================================================================
# DEMO CLASS
# ============================================================================

class EmotionDemo:
    def __init__(self, model_path, device='cpu'):
        """Initialize demo"""
        self.device = torch.device(device)
        self.sample_rate = 16000
        self.max_length = 5
        
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
        
        # Emotion colors for visualization
        self.emotion_colors = {
            'angry': '#FF4444',
            'calm': '#7FB3D5',
            'disgust': '#9B59B6',
            'fearful': '#F39C12',
            'happy': '#FFD700',
            'neutral': '#95A5A6',
            'sad': '#3498DB',
            'surprised': '#FF69B4'
        }
        
        print("Loading model...")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = EmotionRecognitionModel(num_labels=len(self.id_to_emotion))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✓ Model loaded successfully!")
    
    def load_audio(self, audio_path):
        """Load audio file"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr
    
    def predict_with_visualization(self, audio_path, show_plots=True):
        """Predict emotion with detailed visualizations"""
        print(f"\n{'='*80}")
        print(f"Analyzing: {audio_path}")
        print(f"{'='*80}")
        
        # Load audio
        audio, sr = self.load_audio(audio_path)
        duration = len(audio) / sr
        print(f"\nAudio Info:")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sample Rate: {sr} Hz")
        print(f"  Samples: {len(audio)}")
        
        # Show visualizations
        if show_plots:
            print("\nGenerating visualizations...")
            plot_waveform(audio, sr, f"Waveform - {audio_path}")
            plot_spectrogram(audio, sr, f"Mel Spectrogram - {audio_path}")
            plot_mfcc(audio, sr, f"MFCC Features - {audio_path}")
        
        # Preprocess for model
        max_samples = self.sample_rate * self.max_length
        if len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        else:
            audio = audio[:max_samples]
        
        # Process and predict
        print("\nPredicting emotion...")
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_id].item()
        
        predicted_emotion = self.id_to_emotion[predicted_id]
        
        # Get all probabilities
        all_probs = {
            self.id_to_emotion[i]: float(probabilities[0][i])
            for i in range(len(self.id_to_emotion))
        }
        
        # Display results
        print(f"\n{'='*80}")
        print("PREDICTION RESULTS")
        print(f"{'='*80}")
        print(f"\n🎭 Predicted Emotion: {predicted_emotion.upper()}")
        print(f"📊 Confidence: {confidence:.2%}")
        
        print(f"\n{'All Emotion Probabilities:'}")
        print(f"{'-'*80}")
        for emotion, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(prob * 50)
            print(f"  {emotion:12s}: {prob:6.2%} {bar}")
        
        # Plot probabilities
        if show_plots:
            plot_emotion_probabilities(all_probs, f"Emotion Analysis - {audio_path}")
        
        return {
            'emotion': predicted_emotion,
            'confidence': confidence,
            'all_probabilities': all_probs
        }
    
    def compare_multiple_audios(self, audio_paths):
        """Compare emotions across multiple audio files"""
        print(f"\n{'='*80}")
        print(f"COMPARING {len(audio_paths)} AUDIO FILES")
        print(f"{'='*80}")
        
        results = []
        for audio_path in audio_paths:
            result = self.predict_with_visualization(audio_path, show_plots=False)
            result['file'] = audio_path
            results.append(result)
        
        # Create comparison visualization
        emotions = [r['emotion'] for r in results]
        confidences = [r['confidence'] for r in results]
        files = [r['file'].split('/')[-1] for r in results]
        
        plt.figure(figsize=(12, 6))
        colors = [self.emotion_colors.get(e, '#95A5A6') for e in emotions]
        bars = plt.bar(files, confidences, color=colors, alpha=0.7)
        
        # Add labels
        for i, (bar, emotion, conf) in enumerate(zip(bars, emotions, confidences)):
            plt.text(i, conf + 0.02, emotion, ha='center', fontweight='bold')
            plt.text(i, conf - 0.05, f'{conf:.1%}', ha='center', color='white', fontweight='bold')
        
        plt.xlabel('Audio Files', fontsize=12)
        plt.ylabel('Confidence', fontsize=12)
        plt.title('Emotion Comparison Across Multiple Files', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return results

# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run interactive demo"""
    import sys
    
    print("="*80)
    print("AUDIO EMOTION RECOGNITION - INTERACTIVE DEMO")
    print("="*80)
    
    # Check if model exists
    model_path = 'emotion_model.pth'
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model file not found: {model_path}")
        print("Please train the model first using audio_emotion_recognition.py")
        return
    
    # Initialize demo
    demo = EmotionDemo(model_path, device='cpu')
    
    # Example usage
    print("\n" + "="*80)
    print("DEMO OPTIONS")
    print("="*80)
    print("1. Analyze single audio file")
    print("2. Compare multiple audio files")
    print("3. Batch analysis of directory")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        audio_path = input("Enter path to audio file (.wav): ").strip()
        if os.path.exists(audio_path):
            demo.predict_with_visualization(audio_path, show_plots=True)
        else:
            print(f"❌ File not found: {audio_path}")
    
    elif choice == '2':
        num_files = int(input("How many files to compare? "))
        audio_paths = []
        for i in range(num_files):
            path = input(f"Enter path to audio file {i+1}: ").strip()
            if os.path.exists(path):
                audio_paths.append(path)
            else:
                print(f"⚠️  File not found: {path}, skipping...")
        
        if audio_paths:
            demo.compare_multiple_audios(audio_paths)
    
    elif choice == '3':
        dir_path = input("Enter directory path: ").strip()
        if os.path.isdir(dir_path):
            audio_files = [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.endswith('.wav')
            ]
            print(f"\nFound {len(audio_files)} .wav files")
            
            if audio_files:
                for audio_file in audio_files:
                    demo.predict_with_visualization(audio_file, show_plots=False)
        else:
            print(f"❌ Directory not found: {dir_path}")
    
    else:
        print("Invalid choice!")

# Simple usage example
def simple_example():
    """Simple usage example"""
    # Initialize
    demo = EmotionDemo('emotion_model.pth')
    
    # Predict single file
    result = demo.predict_with_visualization('sample_audio.wav')
    print(f"Detected emotion: {result['emotion']}")
    
    # Compare multiple files
    files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
    results = demo.compare_multiple_audios(files)

if __name__ == "__main__":
    import os
    main()