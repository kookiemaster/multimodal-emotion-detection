# Model Evaluation Framework

This document outlines the evaluation framework for the Hybrid Attention Networks (MER-HAN) model for multimodal emotion detection using audio and text.

## Evaluation Metrics

The following metrics are used to evaluate the model's performance:

1. **Accuracy**: The percentage of correctly classified emotions.
2. **F1 Score**: The harmonic mean of precision and recall, calculated for each emotion class and then averaged (weighted by class frequency).
3. **Confusion Matrix**: A table showing the distribution of predicted vs. actual emotion classes.
4. **Per-Class Metrics**:
   - Precision: The ratio of true positives to all predicted positives for each emotion class.
   - Recall: The ratio of true positives to all actual positives for each emotion class.
   - F1 Score: The harmonic mean of precision and recall for each emotion class.

## Evaluation Procedure

The evaluation procedure consists of the following steps:

1. **Data Preparation**:
   - Split the IEMOCAP dataset into train, validation, and test sets.
   - Extract audio features (MFCCs) and tokenize text data.
   - Ensure no speaker overlap between splits to test generalization.

2. **Model Training**:
   - Train the model on the training set.
   - Use validation set for early stopping and hyperparameter tuning.
   - Save the best model based on validation accuracy.

3. **Model Evaluation**:
   - Evaluate the best model on the test set.
   - Calculate all evaluation metrics.
   - Generate visualizations of confusion matrix and attention weights.

4. **Attention Analysis**:
   - Analyze attention weights to understand which parts of the audio and text the model focuses on.
   - Visualize cross-modal attention to understand how the model integrates information from both modalities.

5. **Error Analysis**:
   - Identify emotion classes with poor performance.
   - Analyze misclassified examples to understand model limitations.

## Expected Performance

Based on the literature review and analysis of similar models, we expect the following performance:

1. **Overall Accuracy**: 60-70% on the IEMOCAP dataset.
2. **Per-Class Performance**:
   - Better performance for "angry" and "neutral" emotions.
   - Moderate performance for "sad" and "frustrated" emotions.
   - Lower performance for "happy" and "excited" emotions due to their similar acoustic and linguistic characteristics.

3. **Attention Mechanism Effectiveness**:
   - Local intra-modal attention should highlight emotionally salient parts of each modality.
   - Cross-modal attention should show meaningful relationships between audio and text features.
   - Global inter-modal attention should effectively integrate information from both modalities.

## Comparison with Baselines

The model's performance will be compared with the following baselines:

1. **Unimodal Models**:
   - Text-only emotion recognition using BERT/RoBERTa.
   - Audio-only emotion recognition using CNN/RNN architectures.

2. **Simple Fusion Methods**:
   - Early fusion (feature concatenation).
   - Late fusion (decision-level integration).

3. **Other Multimodal Approaches**:
   - Models without attention mechanisms.
   - Models with only one type of attention mechanism.

## Implementation Notes

The evaluation framework is implemented in the `train.py` and `test_model.py` scripts:

- `train.py` includes the `evaluate_model` function that calculates all metrics and generates visualizations.
- `test_model.py` provides functions for testing the model with sample data and visualizing attention weights.

Due to disk space limitations in the current environment, full evaluation could not be performed. However, the framework is ready to be used once the dependencies are properly installed and the IEMOCAP dataset is available.

## Future Improvements

Based on the evaluation results, the following improvements could be considered:

1. **Model Architecture**:
   - Adjust the attention mechanism parameters.
   - Experiment with different encoder architectures.
   - Add more layers or increase model capacity.

2. **Training Procedure**:
   - Implement data augmentation for underrepresented emotion classes.
   - Use weighted loss functions to address class imbalance.
   - Experiment with different learning rate schedules.

3. **Feature Extraction**:
   - Try different audio features (e.g., spectrograms, prosodic features).
   - Experiment with different text tokenization strategies.
   - Add speaker-specific features to capture individual speaking styles.
