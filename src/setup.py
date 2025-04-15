import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import yaml
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/setup.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_environment():
    """Check if the environment is properly set up."""
    logger.info("Checking environment setup...")
    
    # Check PyTorch installation
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Check transformers installation
    try:
        from transformers import __version__ as transformers_version
        logger.info(f"Transformers version: {transformers_version}")
        
        # Test loading a pre-trained model
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        logger.info("Successfully loaded pre-trained model from Hugging Face")
    except Exception as e:
        logger.error(f"Error loading transformers: {e}")
    
    # Check audio libraries
    try:
        import librosa
        import torchaudio
        logger.info(f"Librosa version: {librosa.__version__}")
        logger.info(f"Torchaudio version: {torchaudio.__version__}")
        logger.info("Audio processing libraries successfully loaded")
    except Exception as e:
        logger.error(f"Error loading audio libraries: {e}")
    
    # Check other dependencies
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn import __version__ as sklearn_version
        
        logger.info(f"NumPy version: {np.__version__}")
        logger.info(f"Pandas version: {pd.__version__}")
        logger.info(f"Scikit-learn version: {sklearn_version}")
        logger.info("Data processing libraries successfully loaded")
    except Exception as e:
        logger.error(f"Error loading data processing libraries: {e}")
    
    logger.info("Environment check completed")

if __name__ == "__main__":
    check_environment()
