# Rock-Paper-Scissors Hand Gesture Recognition

Deep learning-based Rock-Paper-Scissors hand gesture recognition using a custom CNN and MobileNetV2, deployed in a real-time game with a smart AI opponent.

---

## Project Overview

This project implements a deep learning-based hand gesture recognition system for the game Rock-Paper-Scissors. The primary objective was to compare a custom Convolutional Neural Network (CNN) built from scratch with a transfer learning approach using MobileNetV2, and then deploy the trained models in a real-time interactive game with a Smart AI opponent.

The project was developed as part of an AI & Machine Learning coursework and follows the assessment brief requirements, including dataset creation, model training, performance evaluation, and additional AI capabilities.

---

## Results Summary

| Model | Test Accuracy | Test Loss |
|-------|---------------|-----------|
| Custom CNN | 85.51% | 0.3495 |
| MobileNetV2 (Frozen) | 97.10% | 0.1179 |
| MobileNetV2 (Fine-tuned) | 97.10% | 0.1102 |

Transfer learning improved test accuracy by **11.6%** compared to training a CNN from scratch.

---

## File Structure
```
RPS-AICoursework/
│
├── RPS_AICoursework.ipynb
├── rps_game.py
├── README.md
│
├── Models/
│   ├── cnn_model.h5
│   ├── transfer_model_frozen.h5
│   └── transfer_model_finetuned.h5
│
├── Dataset/
│   ├── None/
│   ├── Paper/
│   ├── Rock/
│   └── Scissors/
│
└── Results/
    ├── cnn_confusion_matrix.png
    ├── transfer_confusion_matrix.png
    ├── finetuned_confusion_matrix.png
    ├── cnn_training_accuracy_loss.png
    ├── transfer_training_accuracy_loss.png
    ├── cnn_model_architecture.png
    ├── transfer_model_architecture.png
    ├── cnn_misclassified_examples.png
    ├── transfer_misclassified_examples.png
    ├── test_accuracy_comparison.png
    ├── model_comparison.csv
    ├── results_summary.csv
    ├── final_results_summary.csv
    └── efficiency_summary.csv
```

---

## Installation and Dependencies

### Required Software

- Python 3.8 or higher
- Webcam (for real-time gameplay)
- Google Colab account (for training)

### Required Libraries
```bash
pip install tensorflow==2.13.0
pip install keras==2.13.0
pip install opencv-python==4.8.0
pip install numpy==1.24.0
pip install pandas==2.0.0
pip install matplotlib==3.7.0
pip install seaborn==0.12.0
pip install scikit-learn==1.3.0
```

---

## Usage Instructions

### Training the Models (Google Colab)

1. **Open Google Colab**
   - Go to https://colab.research.google.com/
   - Upload `RPS_AICoursework.ipynb`

2. **Mount Google Drive**
```python
   drive.mount('/content/drive')
```

3. **Upload dataset to Google Drive**
   - Dataset link: https://drive.google.com/drive/folders/1nuhbtqLB8CnJJNWh4VwXWzzGG5e2Iytj

4. **Create the following folder structure in Drive:**
```
   RPS-AICoursework/
   ├── Dataset/
   │   ├── None/
   │   ├── Paper/
   │   ├── Rock/
   │   └── Scissors/
   ├── Models/
   └── Results/
```

5. **Update paths in the notebook and run all cells in order**
   - Training takes approximately 30 minutes
   - Models are saved automatically
   - Download all trained `.h5` files for gameplay

---

### Playing the Game

#### Setup

1. Ensure webcam is connected
2. Place all `.h5` files inside a `Models/` folder next to `rps_game.py`

#### Run the Game
```bash
python rps_game.py
```

#### Model Selection

- Custom CNN (85.51%)
- MobileNetV2 Frozen (97.10%)
- MobileNetV2 Fine-tuned (97.10%)

#### Controls

- **SPACE** – Capture gesture and play a round
- **Q** – Quit the game

---

## Dataset Details

- **Total images:** 460
- **Classes:** Rock, Paper, Scissors, None
- **Images per class:** 115
- **Participants:** 3 
- **Image size:** 224×224 pixels
- **Images captured with varied backgrounds**

**Dataset access:** https://drive.google.com/drive/folders/1nuhbtqLB8CnJJNWh4VwXWzzGG5e2Iytj

---

## Models

### Custom CNN

- 3 convolutional blocks (32, 64, 128 filters)
- Dropout regularisation
- Dense layer (128 neurons)
- 12.9M parameters
- **Test Accuracy:** 85.51%

### MobileNetV2 (Frozen)

- ImageNet pretrained
- Base model frozen
- Custom classification head
- **Test Accuracy:** 97.10%

### MobileNetV2 (Fine-Tuned)

- Top 25% layers unfrozen
- Learning rate: 1e-5
- **Test Accuracy:** 97.10%

---

## Smart AI

The game includes an intelligent AI opponent that learns from player behaviour by:

- Tracking player move history
- Analysing recent patterns using frequency analysis
- Predicting the most likely next move
- Playing the counter-move strategically

This demonstrates adaptive decision-making beyond basic image classification.

---

## Key Findings

- Transfer learning improved accuracy by 11.6%
- MobileNetV2 generalised better than the custom CNN
- Fine-tuning did not improve performance for this dataset size
- Frozen MobileNetV2 was most reliable in real-time gameplay
- Smart AI demonstrated pattern learning and strategic behaviour
