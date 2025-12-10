"""
Audio Emotion Recognition using Pretrained Deep Learning Models
Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Model: Wav2Vec2 (Facebook AI)
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    DATA_PATH = "path/to/ravdess/dataset"  # Update this path
    MODEL_NAME = "facebook/wav2vec2-base"
    SAVE_PATH = "emotion_model.pth"
    
    # Audio parameters
    SAMPLE_RATE = 16000
    MAX_LENGTH = 5  # seconds
    
    # Training parameters
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    
    # Emotion labels (RAVDESS dataset)
    EMOTIONS = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    NUM_CLASSES = len(EMOTIONS)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def parse_ravdess_filename(filepath):
    """
    Parse RAVDESS filename to extract emotion label
    Filename format: 03-01-06-01-02-01-12.wav
    Position 3 = emotion (01-08)
    """
    filename = os.path.basename(filepath)
    emotion_code = filename.split('-')[2]
    return Config.EMOTIONS.get(emotion_code, 'unknown')

def load_audio(filepath, sr=Config.SAMPLE_RATE, max_length=Config.MAX_LENGTH):
    """Load and preprocess audio file"""
    try:
        # Load audio
        audio, sample_rate = librosa.load(filepath, sr=sr)
        
        # Pad or truncate to fixed length
        max_samples = sr * max_length
        if len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        else:
            audio = audio[:max_samples]
        
        return audio
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def create_dataset_dataframe(data_path):
    """Create DataFrame with all audio files and their emotions"""
    data = []
    
    # RAVDESS structure: Actor_XX/*.wav
    for actor_folder in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        
        for audio_file in os.listdir(actor_path):
            if audio_file.endswith('.wav'):
                filepath = os.path.join(actor_path, audio_file)
                emotion = parse_ravdess_filename(filepath)
                
                if emotion != 'unknown':
                    data.append({
                        'filepath': filepath,
                        'emotion': emotion
                    })
    
    df = pd.DataFrame(data)
    
    # Create label encoding
    emotion_labels = sorted(df['emotion'].unique())
    emotion_to_id = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
    id_to_emotion = {idx: emotion for emotion, idx in emotion_to_id.items()}
    
    df['label'] = df['emotion'].map(emotion_to_id)
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Emotion distribution:\n{df['emotion'].value_counts()}")
    
    return df, emotion_to_id, id_to_emotion

# ============================================================================
# PYTORCH DATASET
# ============================================================================

class AudioEmotionDataset(Dataset):
    def __init__(self, dataframe, processor, augment=False):
        self.data = dataframe.reset_index(drop=True)
        self.processor = processor
        self.augment = augment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load audio
        audio = load_audio(row['filepath'])
        
        if audio is None:
            # Return dummy data if loading fails
            audio = np.zeros(Config.SAMPLE_RATE * Config.MAX_LENGTH)
        
        # Apply augmentation (optional)
        if self.augment:
            audio = self.augment_audio(audio)
        
        # Process audio using Wav2Vec2 processor
        inputs = self.processor(
            audio,
            sampling_rate=Config.SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        
        return {
            'input_values': inputs.input_values.squeeze(0),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }
    
    def augment_audio(self, audio):
        """Simple audio augmentation"""
        # Add random noise
        if np.random.random() < 0.5:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
        
        # Time stretching
        if np.random.random() < 0.5:
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        return audio

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_labels, pretrained_model_name=Config.MODEL_NAME):
        super(EmotionRecognitionModel, self).__init__()
        
        # Load pretrained Wav2Vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        
        # Freeze some layers (optional - for faster training)
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # Unfreeze last 2 transformer layers
        for param in self.wav2vec2.encoder.layers[-2:].parameters():
            param.requires_grad = True
        
        # Classification head
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
        # Extract features using Wav2Vec2
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        
        # Global average pooling
        pooled = torch.mean(hidden_states, dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_values = batch['input_values'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_values)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'acc': 100 * correct / total
        })
    
    return total_loss / len(dataloader), 100 * correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_values)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (total_loss / len(dataloader), 
            100 * correct / total, 
            np.array(all_preds), 
            np.array(all_labels))

def train_model(model, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.SAVE_PATH)
            print(f"✓ Model saved with validation accuracy: {val_acc:.2f}%")
    
    return history

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, id_to_emotion):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    emotions = [id_to_emotion[i] for i in sorted(id_to_emotion.keys())]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_emotion(model, processor, audio_path, id_to_emotion, device):
    """Predict emotion for a single audio file"""
    model.eval()
    
    # Load and preprocess audio
    audio = load_audio(audio_path)
    
    if audio is None:
        return None, None
    
    # Process audio
    inputs = processor(
        audio,
        sampling_rate=Config.SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    
    input_values = inputs.input_values.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_values)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_id = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_id].item()
    
    predicted_emotion = id_to_emotion[predicted_id]
    
    return predicted_emotion, confidence

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("AUDIO EMOTION RECOGNITION SYSTEM")
    print("=" * 80)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")
    
    # Load dataset
    print("\n[1/6] Loading dataset...")
    df, emotion_to_id, id_to_emotion = create_dataset_dataframe(Config.DATA_PATH)
    
    # Split dataset
    print("\n[2/6] Splitting dataset...")
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Load processor
    print("\n[3/6] Loading Wav2Vec2 processor...")
    processor = Wav2Vec2Processor.from_pretrained(Config.MODEL_NAME)
    
    # Create datasets and dataloaders
    print("\n[4/6] Creating data loaders...")
    train_dataset = AudioEmotionDataset(train_df, processor, augment=True)
    val_dataset = AudioEmotionDataset(val_df, processor, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    # Initialize model
    print("\n[5/6] Initializing model...")
    model = EmotionRecognitionModel(num_labels=Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    print("\n[6/6] Training model...")
    history = train_model(model, train_loader, val_loader, Config.EPOCHS, Config.DEVICE)
    
    # Plot results
    print("\nPlotting results...")
    plot_training_history(history)
    
    # Final evaluation
    print("\nFinal evaluation on validation set...")
    model.load_state_dict(torch.load(Config.SAVE_PATH))
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, y_pred, y_true = evaluate(model, val_loader, criterion, Config.DEVICE)
    
    print(f"\nFinal Validation Accuracy: {val_acc:.2f}%")
    
    # Classification report
    emotions = [id_to_emotion[i] for i in sorted(id_to_emotion.keys())]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=emotions))
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, id_to_emotion)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {Config.SAVE_PATH}")
    print("=" * 80)
    
    # Example prediction
    print("\nExample prediction:")
    sample_audio = val_df.iloc[0]['filepath']
    pred_emotion, confidence = predict_emotion(model, processor, sample_audio, id_to_emotion, Config.DEVICE)
    actual_emotion = val_df.iloc[0]['emotion']
    print(f"Audio: {sample_audio}")
    print(f"Predicted: {pred_emotion} (confidence: {confidence:.2%})")
    print(f"Actual: {actual_emotion}")

if __name__ == "__main__":
    main()