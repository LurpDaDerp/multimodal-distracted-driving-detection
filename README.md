# Distracted Driving Detection

Detection of driver distraction using multimodal computer vision with feature extraction and deep learning. (85% acc on subjects outside of training)

## Overview

This project detects distracted driving behaviors by analyzing driver face, hand, and body pose. A transformer-based architecture combines spatial landmarks from MediaPipe with CNN visual features to classify driver attention states.

## Architecture

The model leverages three complementary input streams:

- **MediaPipe Landmarks**: Face, hand, and pose keypoints extracted per-frame
- **CNN Features**: EfficientNet/MobileNet encoders extract visual representations from video frames
- **Transformer**: Temporal fusion layer that integrates spatial landmarks and CNN features for inference

## Training

The training pipeline uses **subject-level** splits to ensure that the model does not overfit, and it allow us to see how well the model actually generalizes for subjects not in training data.

## Dataset

Trained on the 100-Driver Dataset
