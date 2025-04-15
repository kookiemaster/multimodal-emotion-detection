import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class LocalAttention(nn.Module):
    """
    Local intra-modal attention mechanism for capturing important features within a single modality.
    """
    def __init__(self, input_dim, hidden_dim):
        super(LocalAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]
        
        # Create query, key, value projections
        Q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key(x)    # (batch_size, seq_len, hidden_dim)
        V = self.value(x)  # (batch_size, seq_len, hidden_dim)
        
        # Calculate attention scores
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale.to(x.device)
        
        # Apply softmax to get attention weights
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention weights to values
        x = torch.matmul(attention, V)
        
        return x, attention

class AudioEncoder(nn.Module):
    """
    Audio encoder with local attention mechanism for the Audio and Text Encoder (ATE) block.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(AudioEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature extraction layers
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_layers)
        ])
        
        # Local attention mechanism
        self.local_attention = LocalAttention(hidden_dim, hidden_dim)
        
        # Final projection layer
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Apply feature extraction layers
        for layer in self.feature_layers:
            x = layer(x)
        
        # Apply local attention
        x, attention_weights = self.local_attention(x)
        
        # Apply final projection
        x = self.projection(x)
        
        return x, attention_weights

class TextEncoder(nn.Module):
    """
    Text encoder with transformer and local attention mechanism for the Audio and Text Encoder (ATE) block.
    """
    def __init__(self, model_name="distilbert-base-uncased", hidden_dim=768, dropout=0.1):
        super(TextEncoder, self).__init__()
        
        # Load pre-trained transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze transformer parameters (optional, can be fine-tuned if needed)
        # for param in self.transformer.parameters():
        #     param.requires_grad = False
        
        # Local attention mechanism
        self.local_attention = LocalAttention(hidden_dim, hidden_dim)
        
        # Final projection layer
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        
        # Apply local attention
        attended_states, attention_weights = self.local_attention(hidden_states)
        
        # Apply final projection
        x = self.dropout(self.projection(attended_states))
        
        return x, attention_weights

class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention (CMA) block for capturing relationships between audio and text modalities.
    """
    def __init__(self, hidden_dim):
        super(CrossModalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
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
        # audio_features shape: (batch_size, audio_seq_len, hidden_dim)
        # text_features shape: (batch_size, text_seq_len, hidden_dim)
        
        batch_size = audio_features.shape[0]
        
        # Audio-to-Text attention
        audio_q = self.audio_query(audio_features)  # (batch_size, audio_seq_len, hidden_dim)
        text_k = self.text_key(text_features)      # (batch_size, text_seq_len, hidden_dim)
        text_v = self.text_value(text_features)    # (batch_size, text_seq_len, hidden_dim)
        
        # Calculate attention scores
        a2t_energy = torch.matmul(audio_q, text_k.permute(0, 2, 1)) / self.scale.to(audio_features.device)
        
        # Apply softmax to get attention weights
        a2t_attention = F.softmax(a2t_energy, dim=-1)
        
        # Apply attention weights to values
        audio_attended_text = torch.matmul(a2t_attention, text_v)
        
        # Text-to-Audio attention
        text_q = self.text_query(text_features)    # (batch_size, text_seq_len, hidden_dim)
        audio_k = self.audio_key(audio_features)   # (batch_size, audio_seq_len, hidden_dim)
        audio_v = self.audio_value(audio_features) # (batch_size, audio_seq_len, hidden_dim)
        
        # Calculate attention scores
        t2a_energy = torch.matmul(text_q, audio_k.permute(0, 2, 1)) / self.scale.to(text_features.device)
        
        # Apply softmax to get attention weights
        t2a_attention = F.softmax(t2a_energy, dim=-1)
        
        # Apply attention weights to values
        text_attended_audio = torch.matmul(t2a_attention, audio_v)
        
        # Return attended features and attention weights
        return audio_attended_text, text_attended_audio, a2t_attention, t2a_attention

class GlobalInterModalAttention(nn.Module):
    """
    Global Inter-Modal Attention mechanism for the Multimodal Emotion Classification (MEC) block.
    """
    def __init__(self, hidden_dim):
        super(GlobalInterModalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Global attention layers
        self.query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key = nn.Linear(hidden_dim * 2, hidden_dim)
        self.value = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, audio_features, text_features):
        # audio_features shape: (batch_size, audio_seq_len, hidden_dim)
        # text_features shape: (batch_size, text_seq_len, hidden_dim)
        
        # Concatenate features along sequence length dimension
        # This creates a combined sequence of audio and text features
        combined_features = torch.cat([audio_features, text_features], dim=1)
        # combined_features shape: (batch_size, audio_seq_len + text_seq_len, hidden_dim)
        
        # Create query, key, value projections
        Q = self.query(combined_features)  # (batch_size, combined_seq_len, hidden_dim)
        K = self.key(combined_features)    # (batch_size, combined_seq_len, hidden_dim)
        V = self.value(combined_features)  # (batch_size, combined_seq_len, hidden_dim)
        
        # Calculate attention scores
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale.to(combined_features.device)
        
        # Apply softmax to get attention weights
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention weights to values
        attended_features = torch.matmul(attention, V)
        
        return attended_features, attention

class EmotionClassifier(nn.Module):
    """
    Final classification layer for emotion prediction.
    """
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

class MERHAN(nn.Module):
    """
    Multimodal Emotion Recognition using Hybrid Attention Networks (MER-HAN).
    """
    def __init__(self, 
                 audio_input_dim=40,  # MFCC features dimension
                 text_model_name="distilbert-base-uncased",
                 hidden_dim=768,
                 num_classes=7,  # 7 emotion classes in IEMOCAP
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
        self.audio_pool = nn.AdaptiveAvgPool1d(1)
        self.text_pool = nn.AdaptiveAvgPool1d(1)
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
        # Transpose for pooling over sequence length dimension
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
