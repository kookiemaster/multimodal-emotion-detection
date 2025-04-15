import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns
import logging
import json
from datetime import datetime

from model import MERHAN
from data import get_iemocap_dataloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class for the MER-HAN model.
    """
    def __init__(self, 
                 model, 
                 train_dataloader, 
                 val_dataloader, 
                 criterion, 
                 optimizer, 
                 scheduler=None, 
                 device='cuda',
                 num_epochs=50,
                 patience=10,
                 checkpoint_dir='checkpoints',
                 log_dir='logs'):
        """
        Initialize the trainer.
        
        Args:
            model: The MER-HAN model
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use for training
            num_epochs: Number of epochs to train for
            patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard', datetime.now().strftime('%Y%m%d-%H%M%S')))
        
        # Move model to device
        self.model.to(device)
        
        # Initialize best validation metrics
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        
    def train_epoch(self, epoch):
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            audio_features = batch['audio_features'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['emotion'].to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = self.model(audio_features, input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / len(self.train_dataloader)
        accuracy = 100. * correct / total
        
        # Log metrics
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/train', accuracy, epoch)
        
        logger.info(f'Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return avg_loss
    
    def validate(self, epoch):
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            tuple: (validation loss, validation accuracy, validation F1 score)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Use tqdm for progress bar
        progress_bar = tqdm(self.val_dataloader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                audio_features = batch['audio_features'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['emotion'].to(self.device)
                
                # Forward pass
                outputs, _ = self.model(audio_features, input_ids, attention_mask)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                total_loss += loss.item()
                
                # Get predictions
                _, predicted = outputs.max(1)
                
                # Store predictions and targets for metrics calculation
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': total_loss / (batch_idx + 1)
                })
        
        # Calculate average loss
        avg_loss = total_loss / len(self.val_dataloader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions) * 100
        f1 = f1_score(all_targets, all_predictions, average='weighted') * 100
        
        # Log metrics
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        self.writer.add_scalar('Accuracy/val', accuracy, epoch)
        self.writer.add_scalar('F1/val', f1, epoch)
        
        logger.info(f'Epoch {epoch+1}/{self.num_epochs} - Val Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1:.2f}%')
        
        # If this is the best validation performance, save the model
        if accuracy > self.best_val_acc:
            logger.info(f'Validation accuracy improved from {self.best_val_acc:.2f}% to {accuracy:.2f}%')
            self.best_val_acc = accuracy
            self.best_val_f1 = f1
            self.best_val_loss = avg_loss
            self.epochs_without_improvement = 0
            
            # Save the model
            self.save_checkpoint(epoch, is_best=True)
        else:
            self.epochs_without_improvement += 1
            logger.info(f'Validation accuracy did not improve. Best: {self.best_val_acc:.2f}%')
            
            # Save regular checkpoint
            self.save_checkpoint(epoch, is_best=False)
        
        return avg_loss, accuracy, f1
    
    def train(self):
        """
        Train the model for the specified number of epochs.
        
        Returns:
            dict: Training history
        """
        logger.info(f'Starting training for {self.num_epochs} epochs')
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        for epoch in range(self.num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate(epoch)
            
            # Update learning rate if scheduler is provided
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning Rate', current_lr, epoch)
                logger.info(f'Learning rate: {current_lr:.6f}')
            
            # Store metrics in history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # Check for early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        logger.info(f'Training completed. Best validation accuracy: {self.best_val_acc:.2f}%')
        
        # Save training history
        self.save_history(history)
        
        # Close TensorBoard writer
        self.writer.close()
        
        return history
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'val_acc': self.best_val_acc,
            'val_f1': self.best_val_f1
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            logger.info(f'Saved best model checkpoint to {best_model_path}')
    
    def save_history(self, history):
        """
        Save training history.
        
        Args:
            history: Training history dictionary
        """
        # Save as JSON
        history_path = os.path.join(self.log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f)
        
        logger.info(f'Saved training history to {history_path}')
        
        # Plot and save training curves
        self.plot_training_curves(history)
    
    def plot_training_curves(self, history):
        """
        Plot and save training curves.
        
        Args:
            history: Training history dictionary
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss curves
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy and F1 curves
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.plot(history['val_f1'], label='Validation F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score (%)')
        ax2.set_title('Validation Metrics')
        ax2.legend()
        ax2.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()

def train_merhan(data_dir, 
                 audio_input_dim=40, 
                 hidden_dim=768, 
                 num_classes=7, 
                 batch_size=16, 
                 num_epochs=50, 
                 learning_rate=1e-4, 
                 weight_decay=1e-5,
                 patience=10,
                 device=None):
    """
    Train the MER-HAN model.
    
    Args:
        data_dir: Directory containing the processed IEMOCAP data
        audio_input_dim: Dimension of audio features
        hidden_dim: Hidden dimension size
        num_classes: Number of emotion classes
        batch_size: Batch size
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate
        weight_decay: Weight decay for L2 regularization
        patience: Patience for early stopping
        device: Device to use for training
        
    Returns:
        tuple: (trained model, training history)
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f'Using device: {device}')
    
    # Create dataloaders
    train_dataloader = get_iemocap_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        split='train',
        num_workers=4
    )
    
    val_dataloader = get_iemocap_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        split='val',
        num_workers=4
    )
    
    # Create model
    model = MERHAN(
        audio_input_dim=audio_input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
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

def evaluate_model(model, test_dataloader, device=None):
    """
    Evaluate the model on the test set.
    
    Args:
        model: Trained MER-HAN model
        test_dataloader: DataLoader for test data
        device: Device to use for evaluation
        
    Returns:
        dict: Evaluation metrics
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f'Evaluating model on device: {device}')
    
    # Move model to device
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    # Use tqdm for progress bar
    progress_bar = tqdm(test_dataloader, desc='Evaluating')
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            audio_features = batch['audio_features'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['emotion'].to(device)
            
            # Forward pass
            outputs, _ = model(audio_features, input_ids, attention_mask)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Store predictions and targets for metrics calculation
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions) * 100
    f1 = f1_score(all_targets, all_predictions, average='weighted') * 100
    
    # Generate classification report
    class_report = classification_report(all_targets, all_predictions, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('logs/confusion_matrix.png')
    plt.close()
    
    # Log results
    logger.info(f'Test Accuracy: {accuracy:.2f}%')
    logger.info(f'Test F1 Score: {f1:.2f}%')
    logger.info(f'Classification Report:\n{classification_report(all_targets, all_predictions)}')
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': class_report,
        'confusion_matrix': cm.tolist()
    }
    
    with open('logs/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

if __name__ == '__main__':
    # Example usage
    data_dir = 'data/processed'
    
    # Train model
    model, history = train_merhan(
        data_dir=data_dir,
        num_epochs=30,
        batch_size=16
    )
    
    # Evaluate model
    test_dataloader = get_iemocap_dataloader(
        data_dir=data_dir,
        batch_size=16,
        split='test',
        num_workers=4
    )
    
    evaluate_model(model, test_dataloader)
