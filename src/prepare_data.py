import os
import pandas as pd
import torch
import torchaudio
import numpy as np
import json
import logging
import argparse
from tqdm import tqdm
import shutil
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_preparation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def extract_emotion_from_evaluation(eval_file_path):
    """
    Extract emotion labels from IEMOCAP evaluation files.
    
    Args:
        eval_file_path (str): Path to the evaluation file
        
    Returns:
        dict: Dictionary mapping utterance IDs to emotion labels
    """
    emotion_dict = {}
    
    try:
        with open(eval_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        utterance_id = parts[0].strip()
                        # Extract the categorical emotion (3rd column in most evaluation files)
                        if len(parts) >= 3:
                            emotion = parts[2].strip().lower()
                        else:
                            emotion = parts[1].strip().lower()
                        
                        # Map to standard emotion categories
                        if emotion in ['exc', 'hap', 'happiness', 'excited']:
                            emotion = 'hap'  # Map excited to happy
                        elif emotion in ['ang', 'angry', 'anger']:
                            emotion = 'ang'
                        elif emotion in ['sad', 'sadness']:
                            emotion = 'sad'
                        elif emotion in ['neu', 'neutral']:
                            emotion = 'neu'
                        elif emotion in ['fru', 'frustrated', 'frustration']:
                            emotion = 'fru'
                        elif emotion in ['fea', 'fear']:
                            emotion = 'fea'
                        elif emotion in ['sur', 'surprise', 'surprised']:
                            emotion = 'sur'
                        elif emotion in ['dis', 'disgust', 'disgusted']:
                            emotion = 'dis'
                        else:
                            emotion = 'oth'  # Other
                        
                        emotion_dict[utterance_id] = emotion
    except Exception as e:
        logger.error(f"Error processing evaluation file {eval_file_path}: {e}")
    
    return emotion_dict

def extract_transcriptions(transcription_file_path):
    """
    Extract transcriptions from IEMOCAP transcription files.
    
    Args:
        transcription_file_path (str): Path to the transcription file
        
    Returns:
        dict: Dictionary mapping utterance IDs to transcriptions
    """
    transcription_dict = {}
    current_id = None
    
    try:
        with open(transcription_file_path, 'r') as f:
            for line in f:
                # Check for utterance ID line
                id_match = re.search(r'\[(.+?)\]', line)
                if id_match:
                    current_id = id_match.group(1)
                    # Extract the transcription text that follows the ID
                    text_match = re.search(r'\[.+?\]: (.*)', line)
                    if text_match:
                        transcription_dict[current_id] = text_match.group(1).strip()
    except Exception as e:
        logger.error(f"Error processing transcription file {transcription_file_path}: {e}")
    
    return transcription_dict

def process_iemocap_dataset(iemocap_dir, output_dir):
    """
    Process the IEMOCAP dataset to create a structured dataset for training.
    
    Args:
        iemocap_dir (str): Path to the IEMOCAP_full_release directory
        output_dir (str): Directory to save processed data
        
    Returns:
        pd.DataFrame: DataFrame containing processed data
    """
    logger.info(f"Processing IEMOCAP dataset from {iemocap_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize lists to store data
    data = []
    
    # Process each session
    for session_id in range(1, 6):  # IEMOCAP has 5 sessions
        session_dir = os.path.join(iemocap_dir, f'Session{session_id}')
        
        if not os.path.exists(session_dir):
            logger.warning(f"Session directory {session_dir} not found. Skipping.")
            continue
        
        logger.info(f"Processing Session {session_id}")
        
        # Process evaluation files to get emotion labels
        emotion_dict = {}
        eval_dir = os.path.join(session_dir, 'dialog', 'EmoEvaluation')
        
        for eval_file in os.listdir(eval_dir):
            if eval_file.endswith('.txt'):
                eval_file_path = os.path.join(eval_dir, eval_file)
                session_emotion_dict = extract_emotion_from_evaluation(eval_file_path)
                emotion_dict.update(session_emotion_dict)
        
        # Process transcription files to get text
        transcription_dict = {}
        transcription_dir = os.path.join(session_dir, 'dialog', 'transcriptions')
        
        for trans_file in os.listdir(transcription_dir):
            if trans_file.endswith('.txt'):
                trans_file_path = os.path.join(transcription_dir, trans_file)
                session_transcription_dict = extract_transcriptions(trans_file_path)
                transcription_dict.update(session_transcription_dict)
        
        # Process audio files
        wav_dir = os.path.join(session_dir, 'sentences', 'wav')
        
        for speaker_dir in os.listdir(wav_dir):
            speaker_path = os.path.join(wav_dir, speaker_dir)
            
            if os.path.isdir(speaker_path):
                for wav_file in os.listdir(speaker_path):
                    if wav_file.endswith('.wav'):
                        utterance_id = wav_file[:-4]  # Remove .wav extension
                        
                        # Check if we have emotion and transcription for this utterance
                        if utterance_id in emotion_dict and utterance_id in transcription_dict:
                            emotion = emotion_dict[utterance_id]
                            text = transcription_dict[utterance_id]
                            
                            # Get speaker ID and session
                            speaker = speaker_dir
                            
                            # Get audio file path
                            audio_path = os.path.join(speaker_path, wav_file)
                            
                            # Copy audio file to output directory
                            output_audio_dir = os.path.join(output_dir, 'audio')
                            os.makedirs(output_audio_dir, exist_ok=True)
                            output_audio_path = os.path.join(output_audio_dir, f"{utterance_id}.wav")
                            
                            try:
                                shutil.copy(audio_path, output_audio_path)
                                
                                # Add to data list
                                data.append({
                                    'utterance_id': utterance_id,
                                    'audio_path': output_audio_path,
                                    'text': text,
                                    'emotion': emotion,
                                    'speaker': speaker,
                                    'session': f'Session{session_id}'
                                })
                            except Exception as e:
                                logger.error(f"Error copying audio file {audio_path}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print dataset statistics
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Emotion distribution:\n{df['emotion'].value_counts()}")
    logger.info(f"Session distribution:\n{df['session'].value_counts()}")
    
    return df

def split_dataset(df, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): DataFrame containing processed data
        output_dir (str): Directory to save split data
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Group by session to ensure no speaker overlap between splits
    sessions = df['session'].unique()
    np.random.shuffle(sessions)
    
    # Calculate number of sessions for each split
    n_sessions = len(sessions)
    n_train = int(n_sessions * train_ratio)
    n_val = int(n_sessions * val_ratio)
    
    # Split sessions
    train_sessions = sessions[:n_train]
    val_sessions = sessions[n_train:n_train+n_val]
    test_sessions = sessions[n_train+n_val:]
    
    # Create splits
    train_df = df[df['session'].isin(train_sessions)]
    val_df = df[df['session'].isin(val_sessions)]
    test_df = df[df['session'].isin(test_sessions)]
    
    # Print split statistics
    logger.info(f"Train set: {len(train_df)} samples, {len(train_sessions)} sessions")
    logger.info(f"Validation set: {len(val_df)} samples, {len(val_sessions)} sessions")
    logger.info(f"Test set: {len(test_df)} samples, {len(test_sessions)} sessions")
    
    # Save splits to CSV
    train_df.to_csv(os.path.join(output_dir, 'iemocap_train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'iemocap_val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'iemocap_test.csv'), index=False)
    
    # Save full dataset
    df.to_csv(os.path.join(output_dir, 'iemocap_full.csv'), index=False)
    
    return train_df, val_df, test_df

def extract_audio_features(df, output_dir, feature_type='mfcc', n_mfcc=40):
    """
    Extract audio features from audio files and save them.
    
    Args:
        df (pd.DataFrame): DataFrame containing processed data
        output_dir (str): Directory to save extracted features
        feature_type (str): Type of features to extract ('mfcc' or 'mel')
        n_mfcc (int): Number of MFCC coefficients
        
    Returns:
        pd.DataFrame: DataFrame with added feature paths
    """
    logger.info(f"Extracting {feature_type} features from audio files")
    
    # Create output directory for features
    features_dir = os.path.join(output_dir, f'{feature_type}_features')
    os.makedirs(features_dir, exist_ok=True)
    
    # Add feature path column to DataFrame
    df['feature_path'] = ''
    
    # Extract features for each audio file
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {feature_type} features"):
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(row['audio_path'])
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Extract features
            if feature_type == 'mfcc':
                # Extract MFCCs
                mfcc_transform = torchaudio.transforms.MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=n_mfcc,
                    melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 40}
                )
                features = mfcc_transform(waveform)
                
            elif feature_type == 'mel':
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
                
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")
            
            # Save features
            feature_path = os.path.join(features_dir, f"{row['utterance_id']}.pt")
            torch.save(features, feature_path)
            
            # Update DataFrame
            df.at[idx, 'feature_path'] = feature_path
            
        except Exception as e:
            logger.error(f"Error extracting features from {row['audio_path']}: {e}")
    
    # Save updated DataFrame
    df.to_csv(os.path.join(output_dir, f'iemocap_with_{feature_type}.csv'), index=False)
    
    return df

def main(args):
    """
    Main function to process the IEMOCAP dataset.
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process IEMOCAP dataset
    df = process_iemocap_dataset(args.iemocap_dir, args.output_dir)
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(
        df, 
        args.output_dir, 
        train_ratio=args.train_ratio, 
        val_ratio=args.val_ratio, 
        test_ratio=args.test_ratio, 
        seed=args.seed
    )
    
    # Extract audio features if specified
    if args.extract_features:
        extract_audio_features(
            df, 
            args.output_dir, 
            feature_type=args.feature_type, 
            n_mfcc=args.n_mfcc
        )
    
    logger.info("IEMOCAP dataset processing completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IEMOCAP dataset for emotion recognition")
    parser.add_argument("--iemocap_dir", type=str, required=True, help="Path to IEMOCAP_full_release directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save processed data")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--extract_features", action="store_true", help="Extract audio features")
    parser.add_argument("--feature_type", type=str, default="mfcc", choices=["mfcc", "mel"], help="Type of features to extract")
    parser.add_argument("--n_mfcc", type=int, default=40, help="Number of MFCC coefficients")
    
    args = parser.parse_args()
    
    main(args)
