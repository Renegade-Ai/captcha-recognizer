# Models Directory

This directory stores trained model files and associated metadata.

## Model Files

### Primary Models

- `captcha_model_with_metadata.pth` - Complete model with all metadata (recommended)
- `captcha_model.pth` - Model state dict only
- `captcha_model_full.pth` - Full PyTorch model object

### Backup/Versioned Models

- `captcha_model_v1.0.0.pth` - Version-tagged models
- `captcha_model_backup_YYYYMMDD.pth` - Daily backups

## Model Metadata

Each model should be accompanied by:

- Character classes used for training
- Image dimensions (height × width)
- Training epochs completed
- Model architecture details
- Training dataset information

## Loading Models

```python
from load_model import load_model_with_metadata

# Load model with all metadata
model, metadata = load_model_with_metadata('outputs/models/captcha_model_with_metadata.pth')

# Access metadata
print(f"Characters: {metadata['char_classes']}")
print(f"Dimensions: {metadata['image_height']}x{metadata['image_width']}")
```

## File Size Guidelines

- Keep models under 100MB for GitHub
- Use Git LFS for larger models
- Consider model compression techniques for deployment
