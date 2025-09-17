import logging
import sys
import time

import torch

import config


class ProgressBar:
    """Custom progress bar that works with logging system"""

    def __init__(self, total, description="Progress", width=50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()

    def update(self, step=1):
        self.current += step
        self._display()

    def _display(self):
        if self.total == 0:
            percent = 100
        else:
            percent = (self.current / self.total) * 100

        filled = (
            int(self.width * self.current // self.total)
            if self.total > 0
            else self.width
        )
        bar = "█" * filled + "░" * (self.width - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"{eta:.0f}s"
        else:
            eta_str = "?s"

        rate = self.current / elapsed if elapsed > 0 else 0

        sys.stdout.write(
            f"\r{self.description}: {bar} {percent:.1f}% ({self.current}/{self.total}) [{rate:.1f}it/s, ETA: {eta_str}]"
        )
        sys.stdout.flush()

    def close(self):
        self._display()
        print()  # New line after completion


def single_epoch_train_model(model, train_loader, optimizer):
    """
    Training function that runs one complete epoch of training.

    This function handles the core training loop including:
    - Forward pass through the model
    - Loss calculation using CTC loss
    - Backward pass (gradient computation)
    - Parameter updates via optimizer

    Args:
        model: The neural network model (CaptchaModel)
        data_loader: PyTorch DataLoader providing batched training data
        optimizer: Optimizer for updating model parameters (Adam)

    Returns:
        float: Average training loss for this epoch
    """

    model.train()

    fin_loss = 0.0

    # Real-time progress tracking
    total_batches = len(train_loader)
    progress_bar = ProgressBar(total_batches, "Training", width=40)

    for batch_idx, data in enumerate(train_loader):

        for key, value in data.items():
            data[key] = value.to(config.DEVICE)

        optimizer.zero_grad()

        # model(**data) is equivalent to model(images=data["images"], targets=data["targets"])
        outputs, loss = model(**data)

        # Debug prints commented out to fix progress bar display
        # if batch_idx == 0:
        #     print(
        #         f"Model prediction shape as {outputs.shape} (seq_len, batch, num_classes)"
        #     )
        #     print(f"Loss item {loss.item():.4f}")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        fin_loss += loss.item()

        # Update progress bar
        progress_bar.update()

    # Close progress bar and log completion
    progress_bar.close()
    logger = logging.getLogger("captcha_training")
    logger.info("Training epoch completed!")

    return fin_loss / len(train_loader)


def single_epoch_eval_model(model, train_loader):

    model.eval()

    fin_loss = 0.0
    fin_preds = []

    # Real-time progress tracking
    total_batches = len(train_loader)
    progress_bar = ProgressBar(total_batches, "Validation", width=40)

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, data in enumerate(train_loader):

            for key, value in data.items():
                data[key] = value.to(config.DEVICE)

            # model(**data) is equivalent to model(images=data["images"], targets=data["targets"])
            batch_preds, loss = model(**data)

            fin_loss += loss.item()

            fin_preds.append(batch_preds)

            # Update progress bar
            progress_bar.update()

    # Close progress bar and log completion
    progress_bar.close()
    logger = logging.getLogger("captcha_training")
    logger.info("Validation epoch completed!")

    return fin_preds, fin_loss / len(train_loader)
