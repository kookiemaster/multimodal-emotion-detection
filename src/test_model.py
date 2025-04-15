import os
import torch
import logging
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from model import MERHAN
from data import get_iemocap_dataloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/model_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_sample_batch():
    """
    Create a sample batch of data for testing the model.
    
    Returns:
        dict: Dictionary containing sample batch data
    """
    # Create random audio features (batch_size, seq_len, features)
    batch_size = 4
    audio_seq_len = 300
    audio_features = torch.randn(batch_size, audio_seq_len, 40)
    
    # Create random input IDs and attention mask
    text_seq_len = 128
    input_ids = torch.randint(0, 30000, (batch_size, text_seq_len))
    attention_mask = torch.ones(batch_size, text_seq_len)
    
    # Create random emotion labels
    emotion = torch.randint(0, 7, (batch_size,))
    
    return {
        'audio_features': audio_features,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'emotion': emotion
    }

def test_model_forward_pass(model, device):
    """
    Test the model's forward pass with a sample batch.
    
    Args:
        model: The MER-HAN model
        device: Device to run the model on
        
    Returns:
        bool: True if forward pass is successful, False otherwise
    """
    logger.info("Testing model forward pass with sample batch")
    
    try:
        # Create sample batch
        batch = create_sample_batch()
        
        # Move batch to device
        audio_features = batch['audio_features'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs, attention_weights = model(audio_features, input_ids, attention_mask)
        
        # Check outputs
        logger.info(f"Output shape: {outputs.shape}")
        logger.info(f"Number of attention weight matrices: {len(attention_weights)}")
        
        # Print sample of outputs
        logger.info(f"Sample output logits: {outputs[0]}")
        
        # Check attention weights
        for name, weights in attention_weights.items():
            logger.info(f"{name} shape: {weights.shape}")
        
        logger.info("Forward pass test successful")
        return True
        
    except Exception as e:
        logger.error(f"Error in forward pass test: {e}")
        return False

def visualize_attention_weights(attention_weights, save_dir="logs"):
    """
    Visualize attention weights from the model.
    
    Args:
        attention_weights: Dictionary of attention weights
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for name, weights in attention_weights.items():
        # Convert to numpy for visualization
        weights_np = weights.cpu().numpy()
        
        # For 3D tensors (batch, seq_len, seq_len), use the first sample
        if len(weights_np.shape) == 3:
            weights_np = weights_np[0]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights_np, cmap='viridis')
        plt.title(f"{name}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}_heatmap.png"))
        plt.close()
        
        logger.info(f"Saved visualization for {name}")

def test_model_with_sample_data(model, data_dir, device, batch_size=4):
    """
    Test the model with sample data.
    
    Args:
        model: The MER-HAN model
        data_dir: Directory containing the sample data
        device: Device to run the model on
        batch_size: Batch size for testing
        
    Returns:
        bool: True if test is successful, False otherwise
    """
    logger.info(f"Testing model with sample data from {data_dir}")
    
    try:
        # Create dataloader
        test_dataloader = get_iemocap_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            split='test',
            num_workers=0  # Use 0 for debugging
        )
        
        if len(test_dataloader) == 0:
            logger.error("Test dataloader is empty")
            return False
        
        logger.info(f"Test dataloader created with {len(test_dataloader)} batches")
        
        # Set model to evaluation mode
        model.eval()
        
        # Initialize metrics
        all_predictions = []
        all_targets = []
        
        # Process batches
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
            # Move batch to device
            audio_features = batch['audio_features'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['emotion'].to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs, attention_weights = model(audio_features, input_ids, attention_mask)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Store predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Visualize attention weights for the first batch
            if batch_idx == 0:
                visualize_attention_weights(attention_weights)
            
            # Only process a few batches for quick testing
            if batch_idx >= 2:
                break
        
        # Calculate metrics
        if len(all_predictions) > 0:
            # Create confusion matrix
            cm = confusion_matrix(all_targets, all_predictions)
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join('logs', 'test_confusion_matrix.png'))
            plt.close()
            
            # Print classification report
            report = classification_report(all_targets, all_predictions)
            logger.info(f"Classification Report:\n{report}")
        
        logger.info("Model test with sample data completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing model with sample data: {e}")
        return False

def main(args):
    """
    Main function to test the model.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Create model
    model = MERHAN(
        audio_input_dim=args.audio_input_dim,
        text_model_name=args.text_model_name,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes
    )
    
    # Move model to device
    model.to(device)
    
    # Test forward pass
    if args.test_forward:
        success = test_model_forward_pass(model, device)
        if not success:
            logger.error("Forward pass test failed")
            return
    
    # Create sample data if needed
    if args.create_sample:
        from prepare_test_data import create_sample_test_data
        create_sample_test_data(args.data_dir, args.num_samples)
    
    # Test with sample data
    if args.test_with_data:
        success = test_model_with_sample_data(model, args.data_dir, device, args.batch_size)
        if not success:
            logger.error("Test with sample data failed")
            return
    
    logger.info("Model testing completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the MER-HAN model")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing the processed data")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run the model on")
    parser.add_argument("--audio_input_dim", type=int, default=40, help="Dimension of audio features")
    parser.add_argument("--text_model_name", type=str, default="distilbert-base-uncased", help="Name of the pre-trained text model")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden dimension size")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of emotion classes")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--test_forward", action="store_true", help="Test model forward pass")
    parser.add_argument("--test_with_data", action="store_true", help="Test model with sample data")
    parser.add_argument("--create_sample", action="store_true", help="Create sample test data")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to create")
    
    args = parser.parse_args()
    
    main(args)
