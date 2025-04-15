# Usage Examples: Hybrid Attention Networks (MER-HAN)

This document provides practical examples of how to use the Hybrid Attention Networks (MER-HAN) model for multimodal emotion detection using audio and text.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training the Model](#training-the-model)
4. [Evaluating the Model](#evaluating-the-model)
5. [Using the Model for Inference](#using-the-model-for-inference)
6. [Visualizing Attention Weights](#visualizing-attention-weights)
7. [Fine-tuning on Custom Data](#fine-tuning-on-custom-data)
8. [Command-Line Interface](#command-line-interface)

## Installation

First, clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/kookiemaster/multimodal-emotion-detection.git
cd multimodal-emotion-detection

# Create a conda environment (recommended)
conda env create -f environment.yml
conda activate mer-han

# Alternatively, install dependencies using pip
pip install -r requirements.txt
```

Verify the installation:

```bash
python src/setup.py
```

## Data Preparation

### Using the IEMOCAP Dataset

If you have access to the IEMOCAP dataset, you can process it using the provided script:

```bash
python src/prepare_data.py \
    --iemocap_dir /path/to/IEMOCAP_full_release \
    --output_dir data/processed \
    --extract_features \
    --feature_type mfcc
```

This will:
1. Process the IEMOCAP dataset
2. Extract emotion labels and transcriptions
3. Extract MFCC features from audio files
4. Split the dataset into train, validation, and test sets
5. Save the processed data to the specified output directory

### Using Sample Data

If you don't have access to the IEMOCAP dataset, you can create sample data for testing:

```bash
python src/prepare_test_data.py \
    --output_dir data/processed \
    --create_sample \
    --num_samples 100
```

### Verifying the Data

To verify that the data is properly processed and ready for training:

```bash
python src/prepare_test_data.py \
    --output_dir data/processed \
    --verify
```

## Training the Model

### Basic Training

To train the model with default parameters:

```bash
python src/train.py \
    --data_dir data/processed \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --patience 10
```

### Custom Training

You can customize various training parameters:

```bash
python src/train.py \
    --data_dir data/processed \
    --audio_input_dim 40 \
    --hidden_dim 512 \
    --num_classes 7 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 5e-5 \
    --weight_decay 1e-6 \
    --patience 15 \
    --device cuda
```

### Training with Class Weights

To handle class imbalance, you can use class weights:

```python
import torch
import numpy as np
import pandas as pd
from src.train import train_merhan

# Load class distribution
df = pd.read_csv('data/processed/iemocap_train.csv')
class_counts = df['emotion'].value_counts().to_dict()

# Calculate class weights
total_samples = len(df)
class_weights = {
    emotion: total_samples / (len(class_counts) * count)
    for emotion, count in class_counts.items()
}

# Convert to tensor
weights = torch.tensor([
    class_weights.get(emotion, 1.0)
    for emotion in sorted(class_weights.keys())
], dtype=torch.float)

# Train with class weights
model, history = train_merhan(
    data_dir='data/processed',
    batch_size=16,
    num_epochs=50,
    class_weights=weights
)
```

## Evaluating the Model

### Basic Evaluation

To evaluate a trained model on the test set:

```bash
python src/train.py \
    --data_dir data/processed \
    --evaluate \
    --checkpoint_path checkpoints/best_model.pt
```

### Custom Evaluation

For more detailed evaluation:

```python
import torch
from src.model import MERHAN
from src.data import get_iemocap_dataloader
from src.train import evaluate_model

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MERHAN(
    audio_input_dim=40,
    hidden_dim=768,
    num_classes=7
)

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# Create test dataloader
test_dataloader = get_iemocap_dataloader(
    data_dir='data/processed',
    batch_size=16,
    split='test',
    num_workers=4
)

# Evaluate the model
results = evaluate_model(model, test_dataloader, device)

# Print results
print(f"Accuracy: {results['accuracy']:.2f}%")
print(f"F1 Score: {results['f1_score']:.2f}%")
print("Classification Report:")
for class_name, metrics in results['classification_report'].items():
    if isinstance(metrics, dict):
        print(f"  {class_name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")
```

## Using the Model for Inference

### Single Sample Inference

To perform emotion detection on a single audio-text pair:

```python
import torch
import torchaudio
from transformers import AutoTokenizer
from src.model import MERHAN

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MERHAN(
    audio_input_dim=40,
    hidden_dim=768,
    num_classes=7
)

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Define emotion mapping
emotion_mapping = {
    0: 'angry',
    1: 'happy',
    2: 'sad',
    3: 'neutral',
    4: 'frustrated',
    5: 'fearful',
    6: 'surprised'
}

def predict_emotion(audio_path, text):
    # Extract audio features
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Extract MFCCs
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=40,
        melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 40}
    )
    features = mfcc_transform(waveform)
    
    # Transpose to get (time, features)
    features = features.squeeze(0).transpose(0, 1)
    
    # Pad or truncate to fixed length (300 time steps)
    max_length = 300
    if features.shape[0] < max_length:
        # Pad
        padding = torch.zeros(max_length - features.shape[0], features.shape[1])
        features = torch.cat([features, padding], dim=0)
    else:
        # Truncate
        features = features[:max_length, :]
    
    # Add batch dimension
    features = features.unsqueeze(0).to(device)
    
    # Tokenize text
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Get input_ids and attention_mask
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Forward pass
    with torch.no_grad():
        logits, attention_weights = model(features, input_ids, attention_mask)
    
    # Get prediction
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1).item()
    
    return {
        'emotion': emotion_mapping[prediction],
        'probabilities': {emotion_mapping[i]: prob.item() for i, prob in enumerate(probabilities[0])},
        'attention_weights': attention_weights
    }

# Example usage
result = predict_emotion('path/to/audio.wav', 'This is an example text.')
print(f"Predicted emotion: {result['emotion']}")
print("Emotion probabilities:")
for emotion, prob in result['probabilities'].items():
    print(f"  {emotion}: {prob:.4f}")
```

### Batch Inference

For processing multiple samples efficiently:

```python
import torch
import pandas as pd
from src.model import MERHAN
from src.data import get_iemocap_dataloader

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MERHAN(
    audio_input_dim=40,
    hidden_dim=768,
    num_classes=7
)

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Define emotion mapping
emotion_mapping = {
    0: 'angry',
    1: 'happy',
    2: 'sad',
    3: 'neutral',
    4: 'frustrated',
    5: 'fearful',
    6: 'surprised'
}

def batch_predict(data_dir, split='test', batch_size=16):
    # Create dataloader
    dataloader = get_iemocap_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        split=split,
        num_workers=4
    )
    
    # Initialize lists to store results
    all_utterance_ids = []
    all_predictions = []
    all_ground_truth = []
    
    # Process batches
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            audio_features = batch['audio_features'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits, _ = model(audio_features, input_ids, attention_mask)
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Store results
            all_utterance_ids.extend(batch['utterance_id'])
            all_predictions.extend([emotion_mapping[pred] for pred in predictions])
            all_ground_truth.extend([emotion_mapping[label.item()] for label in batch['emotion']])
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'utterance_id': all_utterance_ids,
        'predicted_emotion': all_predictions,
        'ground_truth_emotion': all_ground_truth,
        'correct': [pred == gt for pred, gt in zip(all_predictions, all_ground_truth)]
    })
    
    # Save results
    results_df.to_csv('results/batch_predictions.csv', index=False)
    
    return results_df

# Example usage
results = batch_predict('data/processed', split='test', batch_size=16)
print(f"Accuracy: {results['correct'].mean() * 100:.2f}%")
```

## Visualizing Attention Weights

To visualize the attention weights and understand how the model focuses on different parts of the input:

```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.model import MERHAN
from transformers import AutoTokenizer

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MERHAN(
    audio_input_dim=40,
    hidden_dim=768,
    num_classes=7
)

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def visualize_attention(audio_features, text, save_dir='visualizations'):
    # Tokenize text
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Get input_ids and attention_mask
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Forward pass
    with torch.no_grad():
        _, attention_weights = model(audio_features.to(device), input_ids, attention_mask)
    
    # Create directory for visualizations
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize text local attention
    text_local_attn = attention_weights['text_local_attention'][0].cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(text_local_attn, cmap='viridis')
    plt.title('Text Local Attention')
    plt.savefig(f'{save_dir}/text_local_attention.png')
    plt.close()
    
    # Visualize audio-to-text attention
    a2t_attn = attention_weights['audio_to_text_attention'][0].cpu().numpy()
    
    # Get non-padding tokens
    non_padding_mask = attention_mask[0].cpu().numpy().astype(bool)
    valid_tokens = [t for t, m in zip(tokens, non_padding_mask) if m]
    
    # Plot only the relevant part of the attention matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(a2t_attn[:, :len(valid_tokens)], cmap='viridis', 
                xticklabels=valid_tokens, yticklabels=False)
    plt.title('Audio-to-Text Attention')
    plt.xlabel('Text Tokens')
    plt.ylabel('Audio Frames')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/audio_to_text_attention.png')
    plt.close()
    
    # Visualize text-to-audio attention
    t2a_attn = attention_weights['text_to_audio_attention'][0].cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(t2a_attn[:len(valid_tokens), :], cmap='viridis',
                yticklabels=valid_tokens, xticklabels=False)
    plt.title('Text-to-Audio Attention')
    plt.ylabel('Text Tokens')
    plt.xlabel('Audio Frames')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/text_to_audio_attention.png')
    plt.close()
    
    # Visualize global attention
    global_attn = attention_weights['global_attention'][0].cpu().numpy()
    plt.figure(figsize=(12, 10))
    sns.heatmap(global_attn, cmap='viridis')
    plt.title('Global Inter-Modal Attention')
    plt.savefig(f'{save_dir}/global_attention.png')
    plt.close()
    
    print(f"Attention visualizations saved to {save_dir}")
```

## Fine-tuning on Custom Data

To fine-tune the model on your own dataset:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.model import MERHAN

# Define custom dataset
class CustomEmotionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_text_length=128, audio_max_length=300):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.audio_max_length = audio_max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Extract audio features
        audio_features = torch.load(item['feature_path'])
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get input_ids and attention_mask
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Get emotion label
        emotion = item['emotion_label']
        
        return {
            'audio_features': audio_features,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'emotion': torch.tensor(emotion, dtype=torch.long),
            'text': item['text'],
            'id': item['id']
        }

# Fine-tuning function
def fine_tune_model(model, train_dataloader, val_dataloader, num_epochs=10, learning_rate=1e-5):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_dataloader:
            # Move batch to device
            audio_features = batch['audio_features'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['emotion'].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(audio_features, input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                audio_features = batch['audio_features'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['emotion'].to(device)
                
                # Forward pass
                outputs, _ = model(audio_features, input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss/len(train_dataloader):.4f}')
        print(f'  Val Loss: {val_loss/len(val_dataloader):.4f}')
        print(f'  Val Accuracy: {100.0*correct/total:.2f}%')
    
    # Save fine-tuned model
    torch.save(model.state_dict(), 'checkpoints/fine_tuned_model.pt')
    
    return model

# Example usage
from transformers import AutoTokenizer

# Load pre-trained model
model = MERHAN(
    audio_input_dim=40,
    hidden_dim=768,
    num_classes=5  # Adjust for your custom dataset
)

# Load pre-trained weights (optional)
checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Create datasets
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
train_dataset = CustomEmotionDataset('custom_data/train.csv', tokenizer)
val_dataset = CustomEmotionDataset('custom_data/val.csv', tokenizer)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# Fine-tune the model
fine_tuned_model = fine_tune_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs=10,
    learning_rate=1e-5
)
```

## Command-Line Interface

The repository includes command-line scripts for common tasks:

### Training

```bash
python src/train.py --help
```

### Testing

```bash
python src/test_model.py --help
```

### Data Preparation

```bash
python src/prepare_data.py --help
```

### Sample Data Creation

```bash
python src/prepare_test_data.py --help
```

## Complete Example Workflow

Here's a complete workflow from data preparation to model evaluation:

```bash
# 1. Clone the repository
git clone https://github.com/kookiemaster/multimodal-emotion-detection.git
cd multimodal-emotion-detection

# 2. Create and activate conda environment
conda env create -f environment.yml
conda activate mer-han

# 3. Prepare data (using IEMOCAP dataset)
python src/prepare_data.py \
    --iemocap_dir /path/to/IEMOCAP_full_release \
    --output_dir data/processed \
    --extract_features

# 4. Train the model
python src/train.py \
    --data_dir data/processed \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 1e-4

# 5. Evaluate the model
python src/train.py \
    --data_dir data/processed \
    --evaluate \
    --checkpoint_path checkpoints/best_model.pt

# 6. Test the model with sample data
python src/test_model.py \
    --data_dir data/processed \
    --test_with_data \
    --test_forward
```

For users without access to the IEMOCAP dataset, use sample data:

```bash
# Create sample data
python src/prepare_test_data.py \
    --output_dir data/processed \
    --create_sample \
    --num_samples 100

# Test the model with sample data
python src/test_model.py \
    --data_dir data/processed \
    --test_with_data \
    --create_sample
```
