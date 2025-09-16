import torch
from tqdm import tqdm

import config


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

    tk0 = tqdm(train_loader)

    for batch_idx, data in enumerate(tk0):

        for key, value in data.items():
            data[key] = value.to(config.DEVICE)

        optimizer.zero_grad()

        # model(**data) is equivalent to model(images=data["images"], targets=data["targets"])
        outputs, loss = model(**data)

        if batch_idx == 0:
            print(
                f"Model prediction shape as {outputs.shape} (seq_len, batch, num_classes)"
            )

            print(f"Loss item {loss.item():.4f}")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        fin_loss += loss.item()

    return fin_loss / len(train_loader)


def single_epoch_eval_model(model, train_loader):

    model.eval()

    fin_loss = 0.0
    fin_preds = []

    tk0 = tqdm(train_loader)

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, data in enumerate(tk0):

            for key, value in data.items():
                data[key] = value.to(config.DEVICE)

            # model(**data) is equivalent to model(images=data["images"], targets=data["targets"])
            batch_preds, loss = model(**data)

            fin_loss += loss.item()

            fin_preds.append(batch_preds)

    return fin_preds, fin_loss / len(train_loader)
