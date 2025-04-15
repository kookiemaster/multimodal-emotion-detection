# Implementation Details: Hybrid Attention Networks (MER-HAN)

This document provides detailed information about the implementation of the Hybrid Attention Networks (MER-HAN) model for multimodal emotion detection using audio and text.

## Model Architecture

The MER-HAN model consists of three main components:

1. **Audio and Text Encoder (ATE)** with local intra-modal attention
2. **Cross-Modal Attention (CMA)** for capturing relationships between modalities
3. **Multimodal Emotion Classification (MEC)** with global inter-modal attention

### Overall Architecture

```
                     ┌─────────────────┐
                     │                 │
Audio Features ─────►│  Audio Encoder  │──┐
                     │  (with Local    │  │
                     │   Attention)    │  │
                     │                 │  │
                     └─────────────────┘  │  ┌─────────────────┐
                                          │  │                 │
                                          ├─►│  Cross-Modal    │
                                          │  │   Attention     │──┐
                                          │  │                 │  │
                     ┌─────────────────┐  │  └─────────────────┘  │  ┌─────────────────┐
                     │                 │  │                        │  │                 │
Text Input IDs ─────►│  Text Encoder   │──┘                        ├─►│  Global         │──► Emotion
                     │  (with Local    │                           │  │  Inter-Modal    │    Prediction
Attention Mask ─────►│   Attention)    │                           │  │  Attention      │
                     │                 │                           │  │                 │
                     └─────────────────┘                           │  └─────────────────┘
                                                                   │
                                                                   │
                     ┌─────────────────────────────────────────────┘
                     │
                     │
                     └─► Attention Weights (for visualization and analysis)
```

### 1. Audio and Text Encoder (ATE)

#### Audio Encoder

The Audio Encoder processes audio features (typically MFCCs) using a series of linear layers followed by a local attention mechanism.

```python
class AudioEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(AudioEncoder, self).__init__()
        
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_layers)
        ])
        
        self.local_attention = LocalAttention(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # Apply feature extraction layers
        for layer in self.feature_layers:
            x = layer(x)
        
        # Apply local attention
        x, attention_weights = self.local_attention(x)
        
        # Apply final projection
        x = self.projection(x)
        
        return x, attention_weights
```

#### Text Encoder

The Text Encoder uses a pre-trained transformer model (e.g., DistilBERT) followed by a local attention mechanism.

```python
class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", hidden_dim=768, dropout=0.1):
        super(TextEncoder, self).__init__()
        
        # Load pre-trained transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Local attention mechanism
        self.local_attention = LocalAttention(hidden_dim, hidden_dim)
        
        # Final projection layer
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Apply local attention
        attended_states, attention_weights = self.local_attention(hidden_states)
        
        # Apply final projection
        x = self.dropout(self.projection(attended_states))
        
        return x, attention_weights
```

#### Local Attention Mechanism

The Local Attention mechanism helps focus on emotionally salient parts of each modality.

```python
class LocalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LocalAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, x):
        # Create query, key, value projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Calculate attention scores
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale.to(x.device)
        
        # Apply softmax to get attention weights
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention weights to values
        x = torch.matmul(attention, V)
        
        return x, attention
```

### 2. Cross-Modal Attention (CMA)

The Cross-Modal Attention mechanism captures relationships between audio and text modalities.

```python
class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModalAttention, self).__init__()
        
        # Audio-to-Text attention
        self.audio_query = nn.Linear(hidden_dim, hidden_dim)
        self.text_key = nn.Linear(hidden_dim, hidden_dim)
        self.text_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Text-to-Audio attention
        self.text_query = nn.Linear(hidden_dim, hidden_dim)
        self.audio_key = nn.Linear(hidden_dim, hidden_dim)
        self.audio_value = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, audio_features, text_features):
        # Audio-to-Text attention
        audio_q = self.audio_query(audio_features)
        text_k = self.text_key(text_features)
        text_v = self.text_value(text_features)
        
        a2t_energy = torch.matmul(audio_q, text_k.permute(0, 2, 1)) / self.scale.to(audio_features.device)
        a2t_attention = F.softmax(a2t_energy, dim=-1)
        audio_attended_text = torch.matmul(a2t_attention, text_v)
        
        # Text-to-Audio attention
        text_q = self.text_query(text_features)
        audio_k = self.audio_key(audio_features)
        audio_v = self.audio_value(audio_features)
        
        t2a_energy = torch.matmul(text_q, audio_k.permute(0, 2, 1)) / self.scale.to(text_features.device)
        t2a_attention = F.softmax(t2a_energy, dim=-1)
        text_attended_audio = torch.matmul(t2a_attention, audio_v)
        
        return audio_attended_text, text_attended_audio, a2t_attention, t2a_attention
```

### 3. Global Inter-Modal Attention (MEC)

The Global Inter-Modal Attention mechanism integrates information from both modalities for final classification.

```python
class GlobalInterModalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(GlobalInterModalAttention, self).__init__()
        
        # Global attention layers
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, audio_features, text_features):
        # Concatenate features along sequence length dimension
        combined_features = torch.cat([audio_features, text_features], dim=1)
        
        # Create query, key, value projections
        Q = self.query(combined_features)
        K = self.key(combined_features)
        V = self.value(combined_features)
        
        # Calculate attention scores
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale.to(combined_features.device)
        
        # Apply softmax to get attention weights
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention weights to values
        attended_features = torch.matmul(attention, V)
        
        return attended_features, attention
```

### 4. Emotion Classifier

The final classification layer predicts emotion categories.

```python
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super(EmotionClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)
```

### 5. Complete MER-HAN Model

The complete MER-HAN model integrates all components.

```python
class MERHAN(nn.Module):
    def __init__(self, 
                 audio_input_dim=40,
                 text_model_name="distilbert-base-uncased",
                 hidden_dim=768,
                 num_classes=7,
                 dropout=0.3):
        super(MERHAN, self).__init__()
        
        # Audio and Text Encoder (ATE) block
        self.audio_encoder = AudioEncoder(audio_input_dim, hidden_dim, dropout=dropout)
        self.text_encoder = TextEncoder(model_name=text_model_name, hidden_dim=hidden_dim, dropout=dropout)
        
        # Cross-Modal Attention (CMA) block
        self.cross_modal_attention = CrossModalAttention(hidden_dim)
        
        # Global Inter-Modal Attention for Multimodal Emotion Classification (MEC) block
        self.global_attention = GlobalInterModalAttention(hidden_dim)
        
        # Pooling layer to get fixed-size representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier
        self.classifier = EmotionClassifier(hidden_dim, hidden_dim // 2, num_classes, dropout=dropout)
        
    def forward(self, audio_features, input_ids, attention_mask):
        # Audio encoding
        audio_encoded, audio_local_attn = self.audio_encoder(audio_features)
        
        # Text encoding
        text_encoded, text_local_attn = self.text_encoder(input_ids, attention_mask)
        
        # Cross-modal attention
        audio_attended_text, text_attended_audio, a2t_attn, t2a_attn = self.cross_modal_attention(
            audio_encoded, text_encoded
        )
        
        # Combine original and attended features
        audio_features = audio_encoded + audio_attended_text
        text_features = text_encoded + text_attended_audio
        
        # Global inter-modal attention
        global_features, global_attn = self.global_attention(audio_features, text_features)
        
        # Pooling to get fixed-size representation
        global_features = global_features.transpose(1, 2)
        global_features = self.global_pool(global_features).squeeze(-1)
        
        # Classification
        logits = self.classifier(global_features)
        
        return logits, {
            'audio_local_attention': audio_local_attn,
            'text_local_attention': text_local_attn,
            'audio_to_text_attention': a2t_attn,
            'text_to_audio_attention': t2a_attn,
            'global_attention': global_attn
        }
```

## Data Processing

### IEMOCAP Dataset

The Interactive Emotional Dyadic Motion Capture (IEMOCAP) dataset is processed using the following steps:

1. **Extract emotion labels** from evaluation files
2. **Extract transcriptions** from transcription files
3. **Process audio files** and extract features (MFCCs)
4. **Split the dataset** into train, validation, and test sets

```python
def process_iemocap_dataset(iemocap_dir, output_dir):
    # Process each session
    for session_id in range(1, 6):  # IEMOCAP has 5 sessions
        # Process evaluation files to get emotion labels
        emotion_dict = extract_emotion_from_evaluation(eval_files)
        
        # Process transcription files to get text
        transcription_dict = extract_transcriptions(transcription_files)
        
        # Process audio files
        for audio_file in audio_files:
            # Check if we have emotion and transcription for this utterance
            if utterance_id in emotion_dict and utterance_id in transcription_dict:
                # Add to dataset
                data.append({
                    'utterance_id': utterance_id,
                    'audio_path': audio_path,
                    'text': text,
                    'emotion': emotion,
                    'speaker': speaker,
                    'session': session
                })
    
    # Create DataFrame and split dataset
    return df
```

### Feature Extraction

Audio features (MFCCs) are extracted using the following process:

```python
def extract_audio_features(audio_path, feature_type='mfcc'):
    # Load audio file
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
    
    # Pad or truncate to fixed length
    if features.shape[0] < max_length:
        # Pad
        padding = torch.zeros(max_length - features.shape[0], features.shape[1])
        features = torch.cat([features, padding], dim=0)
    else:
        # Truncate
        features = features[:max_length, :]
    
    return features
```

## Training Pipeline

The training pipeline includes:

1. **Data loading** with the `IEMOCAPDataset` class
2. **Model training** with the `Trainer` class
3. **Evaluation** with various metrics

### Training Process

```python
def train_merhan(data_dir, audio_input_dim=40, hidden_dim=768, num_classes=7, batch_size=16, 
                 num_epochs=50, learning_rate=1e-4, weight_decay=1e-5, patience=10, device=None):
    # Create dataloaders
    train_dataloader = get_iemocap_dataloader(data_dir=data_dir, batch_size=batch_size, split='train')
    val_dataloader = get_iemocap_dataloader(data_dir=data_dir, batch_size=batch_size, split='val')
    
    # Create model
    model = MERHAN(audio_input_dim=audio_input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        patience=patience
    )
    
    # Train model
    history = trainer.train()
    
    return model, history
```

### Evaluation Process

```python
def evaluate_model(model, test_dataloader, device=None):
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    all_predictions = []
    all_targets = []
    
    # Process batches
    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to device
            audio_features = batch['audio_features'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['emotion'].to(device)
            
            # Forward pass
            outputs, _ = model(audio_features, input_ids, attention_mask)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Store predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions) * 100
    f1 = f1_score(all_targets, all_predictions, average='weighted') * 100
    
    # Generate classification report
    class_report = classification_report(all_targets, all_predictions, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': class_report,
        'confusion_matrix': cm.tolist()
    }
```

## Testing Framework

The testing framework includes:

1. **Forward pass testing** to verify model architecture
2. **Attention weight visualization** to analyze model behavior
3. **Sample data testing** to verify end-to-end functionality

```python
def test_model_forward_pass(model, device):
    # Create sample batch
    batch = create_sample_batch()
    
    # Move batch to device
    audio_features = batch['audio_features'].to(device)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs, attention_weights = model(audio_features, input_ids, attention_mask)
    
    # Check outputs and attention weights
    return True
```

## Directory Structure

The project has the following directory structure:

```
multimodal-emotion-detection/
├── src/
│   ├── model.py              # Model architecture implementation
│   ├── data.py               # Data loading and processing
│   ├── train.py              # Training and evaluation pipeline
│   ├── prepare_data.py       # IEMOCAP dataset preparation
│   ├── prepare_test_data.py  # Test data preparation
│   ├── test_model.py         # Model testing framework
│   └── setup.py              # Environment setup verification
├── data/
│   └── processed/            # Processed IEMOCAP data
├── checkpoints/              # Saved model checkpoints
├── logs/                     # Training logs and visualizations
├── research_summary.md       # Summary of research on state-of-the-art methods
├── model_analysis.md         # Analysis of top models
├── model_selection.md        # Rationale for selecting MER-HAN
├── evaluation_framework.md   # Evaluation metrics and procedures
├── optimization_strategies.md # Potential model optimizations
├── todo.md                   # Project progress tracking
└── README.md                 # Project overview
```

## Dependencies

The implementation requires the following dependencies:

- PyTorch (1.10+)
- Transformers (4.15+)
- torchaudio (0.10+)
- librosa (0.8+)
- scikit-learn (1.0+)
- pandas (1.3+)
- numpy (1.20+)
- matplotlib (3.5+)
- seaborn (0.11+)
- tqdm (4.62+)

## Limitations and Future Work

The current implementation has the following limitations:

1. **Disk Space Requirements**: The model requires significant disk space for dependencies and dataset storage.
2. **Computational Resources**: Training the full model requires substantial GPU resources.
3. **Dataset Availability**: The IEMOCAP dataset requires licensing and is not publicly available.

Future work could include:

1. **Model Compression**: Implementing techniques to reduce model size and computational requirements.
2. **Transfer Learning**: Pre-training on larger datasets before fine-tuning on IEMOCAP.
3. **Real-time Processing**: Adapting the model for real-time emotion detection applications.

## Conclusion

This document provides detailed information about the implementation of the Hybrid Attention Networks (MER-HAN) model for multimodal emotion detection using audio and text. The implementation follows the architecture described in the research literature and includes comprehensive data processing, training, and evaluation components.
