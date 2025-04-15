# Research Summary: State-of-the-Art Multimodal Emotion Detection Methods

## Introduction
This document summarizes the state-of-the-art methods for multimodal emotion detection focusing on audio and text modalities as requested. The research is based on recent academic papers and repositories.

## Top Methods and Models

### 1. Attention-Based Bimodal Emotion Classification
**Source**: [A Simple Attention-Based Mechanism for Bimodal Emotion Classification](https://arxiv.org/pdf/2407.00134)

**Key Features**:
- Novel deep-learning multimodal architecture for text and audio
- State-of-the-art multimodal emotion classifier
- Mid-level data fusion methods for extracting rich features from different unimodal architectures
- Attention mechanism trained and tested on text and speech data
- Finding: Deep learning architectures trained on different types of data (text and speech) outperform architectures trained only on text or speech

### 2. Hybrid Attention Networks (MER-HAN)
**Source**: [Multimodal emotion recognition based on audio and text by using hybrid attention networks](https://www.sciencedirect.com/science/article/abs/pii/S1746809423004858)

**Key Features**:
- Combines three different attention mechanisms:
  - Local intra-modal attention
  - Cross-modal attention
  - Global inter-modal attention
- Audio and Text Encoder (ATE) block with local intra-modal attention
- Cross-Modal Attention (CMA) block to capture shared feature representations
- Multimodal Emotion Classification (MEC) block with global inter-modal attention
- Tested on IEMOCAP and MELD datasets
- Addresses the challenge of effectively fusing multimodal information

### 3. LLM-Based Feature Fusion Approach
**Source**: [Multimodal Emotion Recognition Using Feature Fusion: An LLM-Based Approach](https://ieeexplore.ieee.org/document/10591796/)

**Key Features**:
- Converts audio and facial features to textual representations
- Uses text augmentation techniques to enrich the dataset
- Early fusion approach combining audio and facial textual descriptions
- Fine-tunes DistilRoBERTa for emotion classification
- Innovative approach using LLMs for multimodal emotion recognition

### 4. VAD-Based Two-Stage Approach
**Source**: [EMOD Repository](https://github.com/xdotli/emod)

**Key Features**:
- Two-stage approach:
  1. Stage 1 (Text-to-VAD): Predicts continuous VAD (Valence-Arousal-Dominance) values from text using a fine-tuned transformer model (RoBERTa-base)
  2. Stage 2 (VAD-to-Emotion): Maps VAD values to discrete emotion categories using rule-based approach or machine learning classifier
- Uses IEMOCAP dataset
- End-to-End Emotion Accuracy: ~46.6%
- Better performance for 'angry' emotion, poor for 'happy' and 'sad'

## Datasets

### IEMOCAP Dataset
The Interactive Emotional Dyadic Motion Capture (IEMOCAP) database is one of the most widely used datasets for multimodal emotion recognition research. It contains approximately 12 hours of audiovisual data, including video, speech, motion capture of face, and text transcriptions.

**Access**: Available through the repository at https://github.com/xdotli/emod, which includes scripts to preprocess the IEMOCAP data.

### MELD Dataset
The Multimodal EmotionLines Dataset (MELD) contains multimodal conversations from the TV series Friends, with audio, visual, and textual features.

## Common Approaches to Multimodal Fusion

1. **Feature-level fusion (Early Fusion)**: Directly concatenating features from different modalities
2. **Decision-level fusion (Late Fusion)**: Modeling modalities independently and merging results
3. **Model-level fusion**: Explicitly exploiting correlation among different modalities
4. **Attention-based fusion**: Using attention mechanisms to focus on relevant parts of each modality

## Next Steps
Based on this research, we will analyze these methods in detail to select the most appropriate model for implementation, considering:
- Technical feasibility
- Performance metrics
- Compatibility with audio and text modalities
- Availability of implementation details
- Suitability for the IEMOCAP dataset
