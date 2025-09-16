# Logs Directory

This directory contains training and evaluation logs.

## Log Types

### Training Logs

- `training_YYYYMMDD_HHMMSS.log` - Complete training session logs
- Contains epoch-by-epoch metrics, loss values, and timing information

### Evaluation Logs

- `evaluation_YYYYMMDD_HHMMSS.log` - Model evaluation results
- Batch prediction results and accuracy calculations

### System Logs

- `system_info.log` - Hardware and environment information
- `error_logs.log` - Error messages and debugging information

## Log Format

Standard log entry format:

```
[TIMESTAMP] [LEVEL] [MODULE] MESSAGE
```

Example:

```
[2024-03-16 14:30:22] [INFO] [TRAINING] Epoch 1/7 - Loss: 2.4563, Accuracy: 23.4%
[2024-03-16 14:30:45] [INFO] [VALIDATION] Validation Loss: 2.1234, Accuracy: 28.9%
```

## Retention Policy

- Keep recent logs (last 30 days)
- Archive older logs if needed for analysis
- Large log files are excluded from git commits
