import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import librosa

class IEMOCAPDataset(Dataset):
    """
    Dataset class for the IEMOCAP dataset, handling both audio and text modalities.
    """
    def __init__(self, 
                 data_dir, 
                 split='train', 
                 max_text_length=128, 
                 audio_feature_type='mfcc',
                 audio_max_length=300,
                 tokenizer_name="distilbert-base-uncased",
                 emotion_mapping=None):
        """
        Initialize the IEMOCAP dataset.
        
        Args:
            data_dir (str): Directory containing the IEMOCAP dataset
            split (str): 'train', 'val', or 'test'
            max_text_length (int): Maximum length of text sequences
            audio_feature_type (str): Type of audio features ('mfcc', 'mel', etc.)
            audio_max_length (int): Maximum length of audio features
            tokenizer_name (str): Name of the pre-trained tokenizer
            emotion_mapping (dict): Mapping from emotion labels to indices
        """
        self.data_dir = data_dir
        self.split = split
        self.max_text_length = max_text_length
        self.audio_feature_type = audio_feature_type
        self.audio_max_length = audio_max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Default emotion mapping for IEMOCAP (7 emotions)
        self.emotion_mapping = emotion_mapping or {
            'ang': 0,  # angry
            'hap': 1,  # happy
            'exc': 1,  # excited (mapped to happy)
            'sad': 2,  # sad
            'neu': 3,  # neutral
            'fru': 4,  # frustrated
            'fea': 5,  # fear
            'sur': 6,  # surprise
            'dis': 7,  # disgust
            'oth': 8   # other
        }
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self):
        """
        Load data from the IEMOCAP dataset.
        
        Returns:
            list: List of dictionaries containing data samples
        """
        # This is a placeholder for the actual data loading logic
        # In a real implementation, you would load the data from the IEMOCAP dataset
        
        # For now, we'll create a dummy dataframe to simulate the data structure
        # In practice, this would be loaded from a CSV file or processed from raw IEMOCAP files
        
        # Check if processed data file exists
        processed_file = os.path.join(self.data_dir, f'iemocap_{self.split}.csv')
        
        if os.path.exists(processed_file):
            print(f"Loading processed data from {processed_file}")
            df = pd.read_csv(processed_file)
            
            # Convert dataframe to list of dictionaries
            data = []
            for _, row in df.iterrows():
                data.append({
                    'audio_path': row['audio_path'],
                    'text': row['text'],
                    'emotion': row['emotion'],
                    'speaker': row['speaker'],
                    'session': row['session'],
                    'utterance_id': row['utterance_id']
                })
            
            return data
        else:
            print(f"Processed data file {processed_file} not found. Please run data preprocessing first.")
            return []
    
    def extract_audio_features(self, audio_path):
        """
        Extract audio features from an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            torch.Tensor: Audio features
        """
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Extract features based on the specified type
            if self.audio_feature_type == 'mfcc':
                # Extract MFCCs
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=40,
                    melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 40}
                )
                features = mfcc_transform(waveform)
                
                # Transpose to get (time, features)
                features = features.squeeze(0).transpose(0, 1)
                
            elif self.audio_feature_type == 'mel':
                # Extract Mel spectrogram
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=400,
                    hop_length=160,
                    n_mels=40
                )
                features = mel_transform(waveform)
                
                # Convert to dB scale
                features = torchaudio.transforms.AmplitudeToDB()(features)
                
                # Transpose to get (time, features)
                features = features.squeeze(0).transpose(0, 1)
                
            else:
                raise ValueError(f"Unsupported audio feature type: {self.audio_feature_type}")
            
            # Pad or truncate to fixed length
            if features.shape[0] < self.audio_max_length:
                # Pad
                padding = torch.zeros(self.audio_max_length - features.shape[0], features.shape[1])
                features = torch.cat([features, padding], dim=0)
            else:
                # Truncate
                features = features[:self.audio_max_length, :]
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            # Return zeros as fallback
            return torch.zeros(self.audio_max_length, 40)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a data sample.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing the sample data
        """
        item = self.data[idx]
        
        # Extract audio features
        audio_features = self.extract_audio_features(item['audio_path'])
        
        # Tokenize text
        text = item['text']
        encoding = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get input_ids and attention_mask
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Get emotion label
        emotion = self.emotion_mapping[item['emotion']]
        
        return {
            'audio_features': audio_features,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'emotion': torch.tensor(emotion, dtype=torch.long),
            'text': text,
            'utterance_id': item['utterance_id']
        }

def get_iemocap_dataloader(data_dir, batch_size=16, split='train', num_workers=4, **kwargs):
    """
    Create a DataLoader for the IEMOCAP dataset.
    
    Args:
        data_dir (str): Directory containing the IEMOCAP dataset
        batch_size (int): Batch size
        split (str): 'train', 'val', or 'test'
        num_workers (int): Number of workers for data loading
        **kwargs: Additional arguments for IEMOCAPDataset
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the IEMOCAP dataset
    """
    dataset = IEMOCAPDataset(data_dir=data_dir, split=split, **kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

def prepare_iemocap_data(iemocap_dir, output_dir, splits=[0.8, 0.1, 0.1]):
    """
    Prepare IEMOCAP data for training, validation, and testing.
    
    Args:
        iemocap_dir (str): Directory containing the IEMOCAP dataset
        output_dir (str): Directory to save processed data
        splits (list): Train, validation, and test splits
        
    Returns:
        None
    """
    # This function would process the raw IEMOCAP data and create CSV files
    # for train, validation, and test splits
    
    # For now, this is a placeholder for the actual implementation
    print(f"Processing IEMOCAP data from {iemocap_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Splits: {splits}")
    
    # In a real implementation, you would:
    # 1. Parse the IEMOCAP directory structure
    # 2. Extract audio files and transcriptions
    # 3. Match audio with transcriptions and emotion labels
    # 4. Split the data into train, validation, and test sets
    # 5. Save the processed data as CSV files
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Example of how to save the processed data
    # train_df.to_csv(os.path.join(output_dir, 'iemocap_train.csv'), index=False)
    # val_df.to_csv(os.path.join(output_dir, 'iemocap_val.csv'), index=False)
    # test_df.to_csv(os.path.join(output_dir, 'iemocap_test.csv'), index=False)
    
    print("IEMOCAP data processing completed")
