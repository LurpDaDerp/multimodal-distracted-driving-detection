# Distracted Driving Detection

Detection of driver distraction using multimodal computer vision with feature extraction and deep learning.

## Overview

This project detects distracted driving behaviors by analyzing driver face, hand, and body pose. A transformer-based architecture combines spatial landmarks from MediaPipe with CNN visual features to classify driver attention states.

## Architecture

The model leverages three complementary input streams:

- **MediaPipe Landmarks**: Face, hand, and pose keypoints extracted per-frame
- **CNN Features**: EfficientNet/MobileNet encoders extract visual representations from video frames
- **Transformer**: Temporal fusion layer that integrates spatial landmarks and CNN features for inference

## Dataset

Trained on the 100-Driver Dataset
