import os
import argparse
import logging
import torch
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/test_data.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_sample_test_data(output_dir, num_samples=10):
    """
    Create a small sample test dataset for testing the model implementation.
    This is useful when the actual IEMOCAP dataset is not available.
    
    Args:
        output_dir (str): Directory to save the sample data
        num_samples (int): Number of samples to create
    """
    logger.info(f"Creating sample test data with {num_samples} samples")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create audio features directory
    audio_features_dir = os.path.join(output_dir, 'audio_features')
    os.makedirs(audio_features_dir, exist_ok=True)
    
    # Create sample data
    data = []
    emotions = ['ang', 'hap', 'sad', 'neu', 'fru', 'fea', 'sur']
    
    for i in range(num_samples):
        # Create random audio features (40 MFCC coefficients, 300 time steps)
        audio_features = torch.randn(1, 40, 300)  # (channels, features, time)
        
        # Save audio features
        feature_path = os.path.join(audio_features_dir, f"sample_{i}.pt")
        torch.save(audio_features, feature_path)
        
        # Create sample text
        texts = [
            "I'm really angry about this situation.",
            "I'm so happy today, everything is going well!",
            "I feel sad about what happened yesterday.",
            "I'm just stating the facts, no emotion here.",
            "This is so frustrating, I can't believe it.",
            "I'm scared of what might happen next.",
            "Wow, I'm really surprised by this news!"
        ]
        
        # Select emotion and corresponding text
        emotion_idx = i % len(emotions)
        emotion = emotions[emotion_idx]
        text = texts[emotion_idx]
        
        # Add to data list
        data.append({
            'utterance_id': f"sample_{i}",
            'audio_path': "N/A",  # Not needed as we're using pre-extracted features
            'feature_path': feature_path,
            'text': text,
            'emotion': emotion,
            'speaker': f"speaker_{i % 2 + 1}",
            'session': f"Session_{i % 5 + 1}"
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    train_df = df[:int(num_samples * 0.6)]
    val_df = df[int(num_samples * 0.6):int(num_samples * 0.8)]
    test_df = df[int(num_samples * 0.8):]
    
    train_df.to_csv(os.path.join(output_dir, 'iemocap_train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'iemocap_val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'iemocap_test.csv'), index=False)
    df.to_csv(os.path.join(output_dir, 'iemocap_full.csv'), index=False)
    
    logger.info(f"Sample test data created and saved to {output_dir}")
    logger.info(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples, Test: {len(test_df)} samples")
    
    return df

def verify_iemocap_data(data_dir):
    """
    Verify that the IEMOCAP data is properly processed and ready for training.
    
    Args:
        data_dir (str): Directory containing the processed IEMOCAP data
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    logger.info(f"Verifying IEMOCAP data in {data_dir}")
    
    # Check if CSV files exist
    required_files = ['iemocap_train.csv', 'iemocap_val.csv', 'iemocap_test.csv']
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            logger.error(f"Required file {file_path} not found")
            return False
    
    # Load CSV files
    try:
        train_df = pd.read_csv(os.path.join(data_dir, 'iemocap_train.csv'))
        val_df = pd.read_csv(os.path.join(data_dir, 'iemocap_val.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'iemocap_test.csv'))
        
        # Check if required columns exist
        required_columns = ['utterance_id', 'text', 'emotion']
        for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Required column {col} not found in {name} dataset")
                    return False
        
        # Check if audio features exist (either audio_path or feature_path should be present)
        if 'audio_path' not in train_df.columns and 'feature_path' not in train_df.columns:
            logger.error("Neither audio_path nor feature_path column found in train dataset")
            return False
        
        # If feature_path is present, check if files exist
        if 'feature_path' in train_df.columns:
            for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
                for idx, row in df.iterrows():
                    if not os.path.exists(row['feature_path']):
                        logger.warning(f"Feature file {row['feature_path']} not found for {name} dataset")
        
        # If audio_path is present, check if files exist
        if 'audio_path' in train_df.columns:
            for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
                for idx, row in df.iterrows():
                    if not os.path.exists(row['audio_path']):
                        logger.warning(f"Audio file {row['audio_path']} not found for {name} dataset")
        
        # Print dataset statistics
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Validation set: {len(val_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        if 'emotion' in train_df.columns:
            logger.info(f"Train emotion distribution:\n{train_df['emotion'].value_counts()}")
            logger.info(f"Validation emotion distribution:\n{val_df['emotion'].value_counts()}")
            logger.info(f"Test emotion distribution:\n{test_df['emotion'].value_counts()}")
        
        logger.info("IEMOCAP data verification completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying IEMOCAP data: {e}")
        return False

def main(args):
    """
    Main function to prepare test data.
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.create_sample:
        # Create sample test data
        create_sample_test_data(args.output_dir, args.num_samples)
    else:
        # Verify existing data
        if args.verify:
            verify_iemocap_data(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare test data for IEMOCAP emotion recognition")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save processed data")
    parser.add_argument("--create_sample", action="store_true", help="Create sample test data")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to create")
    parser.add_argument("--verify", action="store_true", help="Verify existing data")
    
    args = parser.parse_args()
    
    main(args)
