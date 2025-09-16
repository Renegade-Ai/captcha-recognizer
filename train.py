import glob
import logging
import os
import time  # For timing epochs
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn import metrics, model_selection, preprocessing

import config
import dataset
import engine
import model
from model import CaptchaModel


def setup_logging(timestamp):
    """
    Set up comprehensive logging for training session

    This function creates a Python logging system that writes messages to BOTH
    the console (terminal) AND a log file simultaneously. This ensures we have
    a permanent record of training progress while still seeing real-time output.

    Args:
        timestamp: Training session timestamp for file naming (e.g., "20240316_143022")

    Returns:
        tuple: (logger object, log_filename string)
               - logger: Use this to write log messages (logger.info("message"))
               - log_filename: Path where logs are being saved
    """

    # STEP 1: Create the directory structure for storing log files
    # This ensures "outputs/logs/" folder exists before we try to write to it
    # exist_ok=True means don't error if directory already exists
    os.makedirs("outputs/logs", exist_ok=True)

    # STEP 2: Define where our log file will be saved
    # Creates unique filename like: "outputs/logs/training_20240316_143022.log"
    # This prevents overwriting previous training session logs
    log_filename = f"outputs/logs/training_{timestamp}.log"

    # STEP 3: Create the main logger object
    # Think of this as the "central hub" that manages all logging
    # "captcha_training" is a unique name to identify our logger
    logger = logging.getLogger("captcha_training")

    # Set the minimum level of messages to log (DEBUG < INFO < WARNING < ERROR)
    # INFO level captures training progress, errors, and important events
    logger.setLevel(logging.INFO)

    # STEP 4: Clean up any existing handlers to avoid duplicate messages
    # If we run training multiple times, this prevents logging the same message
    # multiple times to multiple files/console outputs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # STEP 5: Create FILE HANDLER - writes messages to a log file
    # This handler captures all log messages and saves them to disk
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)  # Only log INFO level and above to file

    # STEP 6: Create CONSOLE HANDLER - writes messages to terminal/console
    # This handler shows log messages in real-time as training progresses
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Only log INFO level and above to console

    # STEP 7: Create MESSAGE FORMATTER - defines how log messages look
    # This determines the format of each log entry:
    # [2024-03-16 14:30:22] [INFO] Your message here
    #  ^timestamp^         ^level^ ^actual message^
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",  # Message template
        datefmt="%Y-%m-%d %H:%M:%S",  # How to format the timestamp
    )

    # STEP 8: Apply the formatter to both handlers
    # This ensures both file and console output have the same format
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # STEP 9: Attach both handlers to the logger
    # Now when we call logger.info("message"), it will:
    # 1. Write to the log file (file_handler)
    # 2. Display on console (console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # STEP 10: Return the configured logger and filename
    # The caller can use 'logger' to write messages and 'log_filename' to know where logs are saved
    return logger, log_filename


def log_system_info(logger):
    """
    Log comprehensive system and environment information at the start of training

    This function captures all the important details about the training environment
    so we can reproduce results later and understand what hardware/software was used.

    Args:
        logger: The logger object created by setup_logging()
    """

    # Create a visual separator to mark the start of a new training session
    logger.info("=" * 60)
    logger.info("CAPTCHA RECOGNITION TRAINING SESSION STARTED")
    logger.info("=" * 60)

    # LOG PYTORCH AND HARDWARE INFORMATION
    # This helps debug issues and understand performance characteristics
    logger.info(
        f"PyTorch Version: {torch.__version__}"
    )  # Important for reproducibility
    logger.info(f"Device: {config.DEVICE}")  # CPU vs CUDA - affects training speed
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")  # GPU support status

    # If GPU is available, log detailed GPU information
    # This helps understand memory limitations and performance expectations
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")  # GPU model name
        logger.info(
            f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )  # Available GPU memory in GB

    # LOG TRAINING CONFIGURATION PARAMETERS
    # These settings directly affect training behavior and results
    logger.info(f"Configuration:")
    logger.info(f"  - Epochs: {config.EPOCHS}")  # How many times to go through all data
    logger.info(
        f"  - Batch Size: {config.BATCH_SIZE}"
    )  # How many images processed at once
    logger.info(
        f"  - Image Size: {config.IMAGE_HEIGHT}x{config.IMAGE_WIDTH}"
    )  # Input dimensions
    logger.info(f"  - Device: {config.DEVICE}")  # Confirm where training will happen


def remove_duplicates(x):
    """
    Remove consecutive duplicate characters from a string.

    This function is essential for CTC (Connectionist Temporal Classification) output processing.
    CTC often produces consecutive duplicates (e.g., "AABBCC" should become "ABC")
    because it predicts character probabilities at each time step.

    Args:
        x (str): Input string that may contain consecutive duplicate characters

    Returns:
        str: String with consecutive duplicates removed

    Example:
        remove_duplicates("HHEELLLLOO") → "HELLO"
        remove_duplicates("A") → "A"
        remove_duplicates("") → ""
    """
    # Handle edge cases: strings with less than 2 characters don't need processing
    if len(x) < 2:
        return x

    fin = ""  # Initialize result string
    for j in x:
        if fin == "":
            # First character: always add it
            fin = j
        else:
            # Check if current character is same as the last added character
            if j == fin[-1]:
                # Skip consecutive duplicates
                continue
            else:
                # Different character: add it to result
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    """
    Convert raw model predictions into human-readable CAPTCHA text.

    This function processes the neural network's output logits and converts them
    back into character sequences. It handles the CTC blank token and applies
    post-processing to clean up the predictions.

    Args:
        preds (torch.Tensor): Raw model predictions with shape (seq_len, batch_size, num_classes)
                             Output from the neural network's final layer
        encoder (sklearn.LabelEncoder): Fitted label encoder used to convert
                                      character indices back to actual characters

    Returns:
        list: List of predicted CAPTCHA strings, one for each sample in the batch

    Example:
        Input logits → "HELLO" (after processing)
    """
    # Step 1: Rearrange tensor dimensions from (seq_len, batch_size, num_classes)
    # to (batch_size, seq_len, num_classes) for easier processing
    preds = preds.permute(1, 0, 2)

    # Step 2: Apply softmax to convert logits to probability distributions
    # Each position now has probabilities summing to 1.0 across all possible characters
    preds = torch.softmax(preds, 2)

    # Step 3: Get the character index with highest probability at each position
    # This converts from probability distributions to discrete character indices
    preds = torch.argmax(preds, 2)

    # Step 4: Move tensor from GPU to CPU and convert to numpy for easier manipulation
    preds = preds.detach().cpu().numpy()

    cap_preds = []  # Store final CAPTCHA predictions for each sample
    # print("preds shape", preds.shape)
    # Step 5: Process each sample in the batch
    for j in range(preds.shape[0]):
        temp = []  # Temporary list to store characters for this sample

        # Step 6: Convert each predicted index back to a character
        for k in preds[j, :]:
            # Subtract 1 because we added 1 during encoding (CTC blank token handling)
            k = k - 1

            if k == -1:
                # k == -1 represents the CTC blank token (no character)
                # Use special symbol "§" as placeholder for blanks
                temp.append("§")
            else:
                # Convert index back to actual character using the fitted encoder
                p = encoder.inverse_transform([k])[0]
                temp.append(p)

        # Step 7: Join characters and remove blank tokens
        tp = "".join(temp).replace("§", "")

        # Step 8: Remove consecutive duplicate characters (CTC post-processing)
        # CTC often predicts repeated characters, so we clean them up
        final_pred = remove_duplicates(tp)
        cap_preds.append(final_pred)

        # Debug first sample
        # if j == 0:
        #     print(
        #         f"   Sample decode: indices→{preds[j, :5]} chars→{''.join(temp[:5])} final→'{final_pred}'"
        #     )

    return cap_preds


def run_training():
    """
    Main training function that orchestrates the entire CAPTCHA model training process

    This function:
    1. Sets up logging to track training progress
    2. Loads and processes the dataset
    3. Creates the model and training components
    4. Runs the training loop with detailed logging
    5. Saves the trained model with comprehensive metadata
    """

    # ===== LOGGING SETUP =====
    # Generate a unique timestamp for this training session
    # Format: YYYYMMDD_HHMMSS (e.g., "20240316_143022")
    # This ensures each training run has a unique identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize the dual logging system (console + file)
    # From this point forward, all logger.info() calls will:
    # 1. Display in the terminal (real-time feedback)
    # 2. Save to outputs/logs/training_TIMESTAMP.log (permanent record)
    logger, log_filename = setup_logging(timestamp)

    # Log comprehensive system and configuration information
    # This captures the environment details for reproducibility
    log_system_info(logger)

    # Log the session details for easy identification
    logger.info(f"Training session started - ID: {timestamp}")
    logger.info(f"Log file: {log_filename}")

    # ===== DATA LOADING AND PREPROCESSING =====

    image_paths = glob.glob(os.path.join(config.IMAGE_DIR, "*.png"))
    # print("Sample image path", image_paths[0])

    targets_original = [
        image_path.split("/")[1].split(".")[0] for image_path in image_paths
    ]
    # print("Sample target", targets_original[0])

    targets = [[c for c in x] for x in targets_original]

    flattened_targets = [c for clist in targets for c in clist]

    le = preprocessing.LabelEncoder()
    le.fit(flattened_targets)
    logger.info(f"All characters used in label encoding: {le.classes_}")

    targets_encoded = le.transform(flattened_targets)
    logger.info(f"Sample encoded target: {targets_encoded[0]}")

    targets_encoded = np.array(targets_encoded)
    targets_encoded = targets_encoded.reshape(-1, 1)

    targets_encoded = [le.transform(target) for target in targets]

    targets_encoded = np.array(targets_encoded)

    # Adding blank for CTC loss calculation
    targets_encoded = targets_encoded + 1

    (
        train_imgs,  # Training image file paths
        test_imgs,  # Testing images file paths
        train_targets,  # Training encoded targets (integers)
        test_targets,  # Testing encoded targets (integers)
        _,  # Training original targets (not needed, so ignored)
        test_targets_orig,  # Testing original targets (strings, for evaluation)
    ) = model_selection.train_test_split(
        image_paths,
        targets_encoded,
        targets_original,
        test_size=0.1,
        random_state=config.RANDOM_STATE,
    )

    logger.info(f"TRAIN/TEST SPLIT:")
    logger.info(f"   Training images: {len(train_imgs)}")
    logger.info(f"   Testing images:  {len(test_imgs)}")
    logger.info(
        f"   Split ratio:     {len(train_imgs)/(len(train_imgs)+len(test_imgs))*100:.1f}% train, {len(test_imgs)/(len(train_imgs)+len(test_imgs))*100:.1f}% test"
    )

    train_dataset = dataset.ClassificationDataset(
        image_path=train_imgs,
        target=train_targets,
        resize=(
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Randomize order each epoch (important for training)
    )

    test_dataset = dataset.ClassificationDataset(
        image_path=test_imgs,
        target=test_targets,
        resize=(
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
        ),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # No shuffling required during evaluations
    )

    model = CaptchaModel(num_chars=len(le.classes_))
    model.to(config.DEVICE)

    logger.info(f"Model created with {len(le.classes_)} character classes")
    logger.info(f"Model moved to device: {config.DEVICE}")

    # IMPROVED: Increased learning rate for better initial learning
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,  # penalizing large weights in neural network.
    )

    logger.info(f"Optimizer: Adam (lr=1e-3, weight_decay=1e-5)")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.8,  # Multiply LR by 0.8 when plateau detected (less aggressive)
        patience=10,  # Wait 15 epochs before reducing LR (more patient)
        min_lr=1e-6,  # Don't let LR go below this threshold
    )

    logger.info(f"Scheduler: ReduceLROnPlateau (factor=0.8, patience=10)")
    logger.info("=" * 60)
    logger.info("STARTING TRAINING LOOP")
    logger.info("=" * 60)

    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()

        train_loss = engine.single_epoch_train_model(model, train_loader, optimizer)
        valid_preds, test_loss = engine.single_epoch_eval_model(model, test_loader)

        valid_captcha_preds = []

        for _, single_pred in enumerate(valid_preds):
            # decode these predictions and add them to the valid_captcha_preds
            current_pred = decode_predictions(single_pred, le)
            valid_captcha_preds.extend(current_pred)

        combined = list(zip(valid_captcha_preds, test_targets_orig))
        logger.info(f"Epoch {epoch+1}/{config.EPOCHS} - Sample predictions:")
        for i, (pred, actual) in enumerate(combined[:5]):  # Log first 5 examples
            status = "✓" if pred == actual else "✗"
            logger.info(f"   {status} '{actual}' → '{pred}'")

        test_dup_rem = [remove_duplicates(c) for c in test_targets_orig]

        accuracy = metrics.accuracy_score(test_dup_rem, valid_captcha_preds)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1}/{config.EPOCHS} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {test_loss:.4f}, "
            f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%), "
            f"Time: {epoch_duration:.1f}s, "
            f"LR: {current_lr:.2e}"
        )

        # Update learning rate based on validation loss
        old_lr = current_lr
        scheduler.step(test_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr != old_lr:
            logger.info(f"Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")

    logger.info("=" * 60)
    logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    # ===== SAVE THE TRAINED MODEL =====

    # Create outputs/models directory if it doesn't exist
    os.makedirs("outputs/models", exist_ok=True)

    # Save state dict with timestamp
    model_save_path = f"outputs/models/captcha_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"✅ Model state dict saved to: {model_save_path}")

    # Save additional metadata for later use
    model_info = {
        "model_state_dict": model.state_dict(),
        "num_chars": len(le.classes_),
        "char_classes": le.classes_.tolist(),
        "image_height": config.IMAGE_HEIGHT,
        "image_width": config.IMAGE_WIDTH,
        "epoch_trained": config.EPOCHS,
        "training_timestamp": timestamp,
        "training_samples": len(train_imgs),
        "test_samples": len(test_imgs),
        "final_accuracy": accuracy,
        "final_train_loss": train_loss,
        "final_val_loss": test_loss,
    }

    # Save with timestamp
    metadata_path = f"outputs/models/captcha_model_with_metadata_{timestamp}.pth"
    torch.save(model_info, metadata_path)
    logger.info(f"✅ Model with metadata saved to: {metadata_path}")

    # Also save as "latest" for easy access
    latest_metadata_path = "outputs/models/captcha_model_with_metadata_latest.pth"
    torch.save(model_info, latest_metadata_path)
    logger.info(f"✅ Latest model saved to: {latest_metadata_path}")

    # Final training summary
    total_training_time = time.time() - epoch_start_time
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY:")
    logger.info(f"   • Total epochs: {config.EPOCHS}")
    logger.info(f"   • Final accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    logger.info(f"   • Final train loss: {train_loss:.4f}")
    logger.info(f"   • Final validation loss: {test_loss:.4f}")
    logger.info(f"   • Training samples: {len(train_imgs)}")
    logger.info(f"   • Test samples: {len(test_imgs)}")
    logger.info(f"   • Character classes: {len(le.classes_)}")
    logger.info(f"   • Session ID: {timestamp}")
    logger.info(f"   • Log file: {log_filename}")
    logger.info("=" * 60)

    print(f"\n🎉 Training completed! Check logs at: {log_filename}")


if __name__ == "__main__":
    run_training()
