import albumentations as A
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import config
from model import CaptchaModel


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

    return cap_preds


def load_model_with_metadata(
    model_path="outputs/models/captcha_model_with_metadata_latest.pth",
):
    """
    Load model with all metadata (best method)

    Args:
        model_path: Path to saved model with metadata

    Returns:
        tuple: (model, metadata_dict)
    """
    # Load the saved data
    checkpoint = torch.load(model_path, map_location=config.DEVICE)

    # Create model with correct architecture
    model = CaptchaModel(num_chars=checkpoint["num_chars"])

    # Load the state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set to evaluation mode
    model.eval()

    # Return model and metadata
    metadata = {
        "num_chars": checkpoint["num_chars"],
        "char_classes": checkpoint["char_classes"],
        "image_height": checkpoint["image_height"],
        "image_width": checkpoint["image_width"],
        "epoch_trained": checkpoint["epoch_trained"],
    }

    return model, metadata


def preprocess_image(
    image_path, target_height=config.IMAGE_HEIGHT, target_width=config.IMAGE_WIDTH
):
    """
    Preprocess a single CAPTCHA image for inference

    Args:
        image_path: Path to image file
        target_height: Target image height
        target_width: Target image width

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    # Apply same preprocessing as training
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    augmentations = A.Compose(
        [
            A.Resize(height=target_height, width=target_width),
            A.Normalize(mean=mean, std=std),
        ]
    )

    # Apply transformations
    augmented = augmentations(image=image_array)
    processed_image = augmented["image"]

    # Convert to tensor and add batch dimension
    # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    image_tensor = torch.tensor(processed_image.transpose(2, 0, 1), dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    return image_tensor


def predict_captcha(model, image_path, char_classes, device=config.DEVICE):
    """
    Predict CAPTCHA text from image

    Args:
        model: Trained CAPTCHA model
        image_path: Path to CAPTCHA image
        char_classes: List of character classes
        device: Device to run inference on

    Returns:
        str: Predicted CAPTCHA text
    """
    # Create label encoder from char_classes
    le = LabelEncoder()
    le.classes_ = np.array(char_classes)

    # Preprocess image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    print("image_tensor found")

    # Run inference
    with torch.no_grad():
        predictions, _ = model(images=image_tensor, targets=None)

    print("predictions found")

    # Decode predictions using your existing decode_predictions function
    # predictions shape: (seq_len, batch_size, num_classes)
    decoded_results = decode_predictions(predictions, le)

    print("decoded_results found")

    # Return the first (and only) prediction since we processed a single image
    predicted_text = decoded_results[0]

    return predicted_text


# Example usage
if __name__ == "__main__":
    print("🔄 Loading saved model...")

    # Method 1: Load with metadata (recommended)
    try:
        model, metadata = load_model_with_metadata()
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model info: {metadata}")

        # Example inference (replace with actual image path)
        # predicted_text = predict_captcha(model, "input/sample.png", metadata['char_classes'])
        # print(f"🎯 Predicted: {predicted_text}")

        print(f"📊 Character classes: {metadata['char_classes']}")
        print(
            f"📏 Expected input size: {metadata['image_height']}x{metadata['image_width']}"
        )

    except FileNotFoundError:
        print("❌ Model file not found. Train the model first!")
