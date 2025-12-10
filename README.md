# Audio Emotion Recognition - Complete Setup Guide

## 📋 Overview
This system uses Facebook's Wav2Vec2 pretrained model for audio emotion recognition from .wav files.

---

## 🔧 Installation

### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python -m venv emotion_env

# Activate environment
# On Linux/Mac:
source emotion_env/bin/activate
# On Windows:
emotion_env\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# For GPU support, use:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers==4.35.0
pip install librosa==0.10.1
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install tqdm==4.66.1
pip install soundfile==0.12.1
```

### requirements.txt
```text
torch>=2.0.0
transformers==4.35.0
librosa==0.10.1
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
tqdm==4.66.1
soundfile==0.12.1
```

---

## 📊 Dataset Setup

### Option 1: RAVDESS (Recommended)

**Download:**
```bash
# Visit: https://zenodo.org/record/1188976
# Download: Audio_Speech_Actors_01-24.zip (unzip to get 24 actor folders)
```

**Directory Structure:**
```
ravdess/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   ├── 03-01-01-01-01-02-01.wav
│   └── ...
├── Actor_02/
│   └── ...
└── Actor_24/
    └── ...
```

**Filename Format:**
- `03-01-06-01-02-01-12.wav`
- Position 3 (06) = Emotion code:
  - 01 = neutral
  - 02 = calm
  - 03 = happy
  - 04 = sad
  - 05 = angry
  - 06 = fearful
  - 07 = disgust
  - 08 = surprised

### Option 2: TESS

**Download:**
```bash
# Visit: https://tspace.library.utoronto.ca/handle/1807/24487
# Download all .zip files and extract
```

### Option 3: Combined Dataset (Best Performance)
Combine RAVDESS + TESS + CREMA-D for better generalization.

---

## 🚀 Quick Start

### 1. Training the Model

```bash
# Update Config.DATA_PATH in the script to your dataset location
python audio_emotion_recognition.py
```

**Configuration (in code):**
```python
class Config:
    DATA_PATH = "/path/to/ravdess/dataset"  # UPDATE THIS
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-4
```

**Expected Output:**
```
Dataset loaded: 1440 samples
Train samples: 1152, Validation samples: 288
Training model...
Epoch 1/20
Train Loss: 1.8234 | Train Acc: 35.42%
Val Loss: 1.6532 | Val Acc: 42.36%
✓ Model saved with validation accuracy: 42.36%
...
Final Validation Accuracy: 78.47%
```

### 2. Using the Trained Model

**Single File Prediction:**
```bash
python emotion_inference.py --audio sample.wav --model emotion_model.pth
```

**Output:**
```
PREDICTION RESULT
================================================================================
Predicted Emotion: happy
Confidence: 87.23%

All Probabilities:
  happy       : 87.23% ███████████████████████████████████████████
  surprised   : 6.45%  ███
  neutral     : 3.21%  █
  calm        : 1.87%  
  ...
```

**Batch Processing:**
```bash
python emotion_inference.py --audio ./audio_folder/ --model emotion_model.pth --batch
```

**Python API Usage:**
```python
from emotion_inference import EmotionPredictor

# Initialize predictor
predictor = EmotionPredictor('emotion_model.pth', device='cpu')

# Predict single file
result = predictor.predict('my_audio.wav')
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = predictor.predict_batch(files)
```

---

## 📈 Expected Performance

### RAVDESS Dataset
- **Training Accuracy:** 85-95%
- **Validation Accuracy:** 75-85%
- **Best Emotions:** Happy, Angry, Sad (>85%)
- **Challenging Emotions:** Neutral, Calm, Disgust (65-75%)

### Training Time
- **CPU:** ~3-4 hours (20 epochs)
- **GPU (Tesla T4):** ~30-45 minutes (20 epochs)

---

## 🎯 Model Architecture

```
Input: Raw Audio Waveform (.wav)
    ↓
Wav2Vec2 Feature Extractor (Pretrained)
    ↓
Wav2Vec2 Transformer Encoder (Partially Fine-tuned)
    ↓
Global Average Pooling
    ↓
Classification Head:
    - Linear (768 → 256)
    - ReLU + Dropout
    - Linear (256 → 128)
    - ReLU + Dropout
    - Linear (128 → 8 emotions)
    ↓
Softmax
    ↓
Output: Emotion Probabilities
```

**Parameters:**
- Total: ~95M parameters
- Trainable: ~2M parameters (last 2 transformer layers + classifier)

---

## 🔬 Preprocessing Steps

1. **Audio Loading**
   - Sample rate: 16kHz (Wav2Vec2 requirement)
   - Duration: Padded/truncated to 5 seconds

2. **Normalization**
   - Handled automatically by Wav2Vec2Processor

3. **Augmentation** (Training only)
   - Random noise addition
   - Time stretching (0.9-1.1x)

---

## 🎨 Customization

### Change Emotions
```python
# Modify Config.EMOTIONS dictionary
EMOTIONS = {
    '01': 'neutral',
    '02': 'happy',
    '03': 'sad',
    # Add your own mappings
}
```

### Adjust Training Parameters
```python
BATCH_SIZE = 32  # Increase for faster training (needs more memory)
EPOCHS = 30      # More epochs for better convergence
LEARNING_RATE = 5e-5  # Lower for fine-tuning
```

### Use Different Pretrained Model
```python
# Larger model for better accuracy
MODEL_NAME = "facebook/wav2vec2-large"

# Smaller model for faster inference
MODEL_NAME = "facebook/wav2vec2-base-960h"
```

---

## 🐛 Troubleshooting

### Out of Memory Error
```python
# Reduce batch size
BATCH_SIZE = 8

# Use gradient accumulation
# Add in training loop:
if (batch_idx + 1) % 2 == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### Low Accuracy
1. **Train longer:** Increase epochs to 30-40
2. **Data augmentation:** Enable more augmentation
3. **Learning rate:** Try 5e-5 or 1e-5
4. **Unfreeze more layers:** Fine-tune more transformer layers

### Audio Loading Issues
```bash
# Install additional audio libraries
pip install ffmpeg-python
sudo apt-get install libsndfile1  # Linux
brew install libsndfile  # Mac
```

---

## 📊 Evaluation Metrics

After training, you'll get:

1. **Training History Plot**
   - Loss curves
   - Accuracy curves

2. **Confusion Matrix**
   - Shows which emotions are confused

3. **Classification Report**
   - Precision, Recall, F1-score per emotion

4. **Sample Predictions**
   - Real examples with confidence scores

---

## 🚀 Deployment Options

### Flask API
```python
from flask import Flask, request, jsonify
from emotion_inference import EmotionPredictor

app = Flask(__name__)
predictor = EmotionPredictor('emotion_model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['audio']
    audio_file.save('temp.wav')
    result = predictor.predict('temp.wav')
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

### Gradio Interface
```python
import gradio as gr
from emotion_inference import EmotionPredictor

predictor = EmotionPredictor('emotion_model.pth')

def predict(audio):
    result = predictor.predict(audio)
    return result['emotion'], result['confidence']

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Label(label="Emotion"), gr.Number(label="Confidence")],
    title="Audio Emotion Recognition"
)

demo.launch()
```

---

## 📚 Additional Resources

- **Wav2Vec2 Paper:** https://arxiv.org/abs/2006.11477
- **RAVDESS Paper:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391
- **Hugging Face Wav2Vec2:** https://huggingface.co/docs/transformers/model_doc/wav2vec2

---

## 💡 Tips for Best Results

1. **Data Quality:** Use high-quality audio recordings (16kHz+, minimal noise)
2. **Balanced Dataset:** Ensure roughly equal samples per emotion
3. **Cross-validation:** Train multiple models with different splits
4. **Ensemble:** Combine predictions from multiple models
5. **Domain-specific:** Fine-tune on your target domain (call center, therapy, etc.)

---

## 📝 Citation

If you use this code in research, please cite:

```bibtex
@misc{wav2vec2_emotion,
  author = {Your Name},
  title = {Audio Emotion Recognition using Wav2Vec2},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/emotion-recognition}}
}
```