# Outputs Directory

This directory contains all the generated outputs from the CAPTCHA recognition model training and inference.

## Directory Structure

```
outputs/
├── models/          # Trained model files (.pth, .pt)
├── logs/            # Training logs and metrics
├── predictions/     # Prediction results and accuracy reports
├── visualizations/  # Plots, charts, and visual analysis
└── checkpoints/     # Model checkpoints during training
```

## File Naming Conventions

### Models

- `captcha_model_YYYYMMDD_HHMMSS.pth` - Timestamp-based model files
- `captcha_model_best.pth` - Best performing model
- `captcha_model_latest.pth` - Most recent model

### Logs

- `training_YYYYMMDD_HHMMSS.log` - Training session logs
- `evaluation_YYYYMMDD_HHMMSS.log` - Model evaluation logs

### Predictions

- `predictions_YYYYMMDD_HHMMSS.json` - Batch prediction results
- `accuracy_report_YYYYMMDD_HHMMSS.txt` - Accuracy analysis

### Visualizations

- `loss_curves_YYYYMMDD.png` - Training/validation loss plots
- `accuracy_plots_YYYYMMDD.png` - Accuracy trend visualizations
- `confusion_matrix_YYYYMMDD.png` - Character-level confusion matrices

## Usage Notes

- Most files in subdirectories are ignored by git (see .gitignore)
- Only small analysis files and reports should be committed
- Large model files (>100MB) should use Git LFS or external storage
- README files in each subdirectory provide specific guidance
