# Rock Paper Scissors AI Classifier
**Cornell University - AI for Engineering Management**
**Elisha Gras | March 2026**

## Overview
Real-time Rock Paper Scissors gesture classifier running on an ESP32-S3 
microcontroller using a custom-trained Convolutional Neural Network (CNN).

## How It Works
1. ESP32 camera captures a 96x96 grayscale image
2. Image is resized to 32x32 and thresholded at 128
3. CNN runs inference on the 32x32 binary image
4. Prediction is printed to the host PC via serial

## Files
- `classify.py` — Real-time classification on ESP32
- `train_cnn.py` — CNN model training on laptop
- `reprocess.py` — Dataset preprocessing and augmentation
- `convert_model.py` — Keras to TFLite conversion
- `prs_cnn.tmdl` — Trained model for ESP32
- `citations.txt` — Code attribution and citations
- `RPS_Project_Documentation.docx` — Full project documentation

## Results
- Training accuracy: 98%
- Dataset: ~680 processed images per class (rock, paper, scissors)
- Model size on ESP32: 103.8 KB flash, 11.0 KB RAM

## Dependencies
- TensorFlow / Keras (laptop training)
- emlearn_cnn_fp32 (ESP32 inference)
- Pillow (image preprocessing)
