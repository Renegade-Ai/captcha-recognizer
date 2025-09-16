# 🔐 Advanced CAPTCHA Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced CAPTCHA recognition system using deep learning with CNN + LSTM/GRU architecture. This project implements state-of-the-art computer vision techniques to automatically solve text-based CAPTCHAs.

## 🌟 Features

- **Hybrid Architecture**: CNN for feature extraction + LSTM for sequence modeling
- **CTC Loss**: Connectionist Temporal Classification for variable-length sequences
- **Data Augmentation**: Robust preprocessing with Albumentations
- **Model Persistence**: Complete model saving/loading with metadata
- **Batch Processing**: Efficient training and inference pipelines
- **Comprehensive Logging**: Detailed training metrics and predictions
- **Easy Deployment**: Ready-to-use prediction interface

## 🏗️ Architecture

```
Input CAPTCHA Image (96×300×3)
           ↓
    Convolutional Layers
    - Conv2D + BatchNorm + ReLU
    - MaxPool2D (height only)
           ↓
    Feature Reshaping
    - Permute dimensions
    - Flatten spatial features
           ↓
    Linear Transformation
    - Compress features
    - Prepare for RNN
           ↓
    Bidirectional LSTM
    - Forward + backward context
    - Character sequence modeling
           ↓
    Output Layer + CTC Loss
    - Character probabilities
    - Sequence alignment
```

## 📁 Project Structure

```
captcha-recognition-adv/
├── 📄 Core Files
│   ├── model.py              # Neural network architecture
│   ├── dataset.py            # Data loading and preprocessing
│   ├── engine.py             # Training and evaluation engine
│   ├── train.py              # Training script
│   ├── config.py             # Configuration settings
│   └── load_model.py         # Model loading and inference
│
├── 🎯 Generation & Testing
│   ├── image_generator.py    # CAPTCHA image generation
│   └── test_prediction.py    # Prediction testing suite
│
├── 📊 Outputs
│   ├── models/               # Trained models and checkpoints
│   ├── logs/                 # Training logs and metrics
│   ├── predictions/          # Prediction results
│   └── visualizations/       # Plots and analysis
│
├── 📚 Documentation
    ├── README.md             # This file
    ├── requirements.txt      # Python dependencies
    └── docs/                 # Additional documentation

```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/captcha-recognition-adv.git
cd captcha-recognition-adv

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
# Generate 30,000 CAPTCHA images
python image_generator.py
```

This creates synthetic CAPTCHA images in the `input/` directory with the format: `{text}.png`

### 3. Train the Model

```bash
# Start training
python train.py
```

Training outputs:

- **Models**: Saved to `outputs/models/` with timestamps
- **Logs**: Training metrics and progress
- **Checkpoints**: Intermediate model states

### 4. Test Predictions

```bash
# Test the trained model
python test_prediction.py
```

## 🔧 Configuration

Edit `config.py` to customize training parameters:

```python
# Training settings
EPOCHS = 7
BATCH_SIZE = 8
IMAGE_HEIGHT = 96  # Match generated images
IMAGE_WIDTH = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data settings
IMAGE_DIR = "input"
RANDOM_STATE = 7
```

## 🧠 Model Architecture Details

### CNN Feature Extractor

- **Conv1**: 3→128 channels, (3×6) kernel
- **Conv2**: 128→64 channels, (3×6) kernel
- **Pooling**: Height-only reduction (preserves width for RNN)
- **BatchNorm**: Stabilizes training
- **Dropout**: Prevents overfitting

### Sequence Processor

- **Linear Layer**: Compresses CNN features (1152→64)
- **Bidirectional LSTM**: Processes sequence in both directions
- **CTC Output**: Handles variable-length character sequences

### Loss Function

- **CTC Loss**: Automatically aligns predictions with targets
- **No explicit segmentation** required
- **Handles variable-length** CAPTCHA texts

## 📊 Performance Metrics

The model tracks:

- **Training/Validation Loss**: CTC loss values
- **Character Accuracy**: Individual character recognition rate
- **Sequence Accuracy**: Complete CAPTCHA match rate
- **Training Time**: Per epoch and total training duration

## 🔮 Usage Examples

### Basic Prediction

```python
from load_model import load_model_with_metadata, predict_captcha

# Load trained model
model, metadata = load_model_with_metadata()

# Predict CAPTCHA text
result = predict_captcha(model, "path/to/captcha.png", metadata['char_classes'])
print(f"Predicted: {result}")
```

### Batch Prediction

```python
import os
from load_model import load_model_with_metadata, predict_captcha

# Load model once
model, metadata = load_model_with_metadata()

# Process multiple images
image_dir = "input"
for filename in os.listdir(image_dir):
    if filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        prediction = predict_captcha(model, image_path, metadata['char_classes'])
        actual = os.path.splitext(filename)[0]
        print(f"{filename}: {actual} → {prediction}")
```

### Custom Model Loading

```python
from load_model import load_model_with_metadata

# Load specific model version
model, metadata = load_model_with_metadata("outputs/models/captcha_model_with_metadata_20240316_143022.pth")

# Access model information
print(f"Model trained for {metadata['epoch_trained']} epochs")
print(f"Character set: {''.join(metadata['char_classes'])}")
print(f"Training samples: {metadata['training_samples']}")
```

## 🛠️ Development

### Training from Scratch

1. **Generate Data**: Create synthetic CAPTCHAs with `image_generator.py`
2. **Configure**: Adjust settings in `config.py`
3. **Train**: Run `python train.py`
4. **Evaluate**: Test with `test_prediction.py`
5. **Iterate**: Adjust architecture/hyperparameters as needed

### Model Customization

The architecture can be easily modified in `model.py`:

- **CNN layers**: Adjust channels, kernel sizes, pooling
- **RNN**: Switch between LSTM/GRU, change hidden sizes
- **Output**: Modify character set or sequence length

### Adding New Features

- **Data Augmentation**: Extend `dataset.py` with new transformations
- **Loss Functions**: Experiment with different loss combinations
- **Metrics**: Add custom evaluation metrics in `engine.py`

## 📈 Results and Analysis

Training typically achieves:

- **Character Accuracy**: 85-95% (depending on CAPTCHA complexity)
- **Sequence Accuracy**: 70-85% (complete CAPTCHA match)
- **Training Time**: ~10-30 minutes on modern hardware

Results are saved to:

- `outputs/logs/` - Training metrics and logs
- `outputs/predictions/` - Prediction results and accuracy reports
- `outputs/visualizations/` - Loss curves and accuracy plots

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch**: Deep learning framework
- **Albumentations**: Data augmentation library
- **CTC Loss**: For sequence alignment
- **Synthetic CAPTCHAs**: Training data generation

## 📚 References

- [CTC: Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [Deep Learning for Computer Vision](https://www.deeplearningbook.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🐛 Issues and Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/captcha-recognition-adv/issues) page
2. Create a new issue with detailed description
3. Include error logs and system information

---

**Made with ❤️ for the computer vision community**
