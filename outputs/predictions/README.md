# Predictions Directory

This directory stores prediction results and accuracy reports.

## File Types

### Prediction Results

- `predictions_YYYYMMDD_HHMMSS.json` - Batch prediction results
- `single_prediction_YYYYMMDD_HHMMSS.txt` - Individual prediction logs

### Accuracy Reports

- `accuracy_report_YYYYMMDD_HHMMSS.txt` - Detailed accuracy analysis
- `confusion_matrix_YYYYMMDD_HHMMSS.csv` - Character-level confusion data

### Comparison Files

- `prediction_comparison_YYYYMMDD.csv` - Actual vs predicted comparisons
- `error_analysis_YYYYMMDD.txt` - Analysis of incorrect predictions

## JSON Format Example

```json
{
  "session_info": {
    "timestamp": "2024-03-16_14:30:22",
    "model_used": "captcha_model_with_metadata_20240316_143022.pth",
    "total_samples": 100
  },
  "predictions": [
    {
      "image": "abc123.png",
      "actual": "abc123",
      "predicted": "abc123",
      "confidence": 0.95,
      "correct": true
    }
  ],
  "summary": {
    "total_correct": 85,
    "total_samples": 100,
    "accuracy": 0.85
  }
}
```

## Usage

These files can be used for:

- Model performance analysis
- Error pattern identification
- Comparative studies between model versions
- Research and development insights
