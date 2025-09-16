import albumentations as A
import numpy as np
import torch
from PIL import Image


class ClassificationDataset:
    """
    PyTorch Dataset class for CAPTCHA image classification.

    This dataset handles loading CAPTCHA images and their corresponding text labels,
    applying preprocessing transformations, and returning data in a format suitable
    for training neural networks with CTC (Connectionist Temporal Classification) loss.
    """

    def __init__(self, image_path, target, resize=None):
        """
        Initialize the dataset with image paths, targets, and optional resizing.

        Args:
            image_paths (list): List of file paths to CAPTCHA images
            targets (list): List of encoded target sequences (character labels for each image)
            resize (tuple, optional): Target size as (height, width) for resizing images
                                    Example: (75, 300) resizes images to 75x300 pixels
        """
        self.image_path = image_path
        self.target = target
        self.resize = resize

        # ImageNet pre-trained model statistics for normalization
        # These values are standard for models pre-trained on ImageNet dataset
        mean = (0.485, 0.456, 0.406)  # RGB channel means
        std = (0.229, 0.224, 0.225)  # RGB channel standard deviations

        self.augmentations = A.Compose([A.Normalize(mean, std, max_pixel_value=255.0)])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        """
        Load and preprocess a single sample from the dataset.

        This method is called by PyTorch's DataLoader to fetch individual samples
        during training/testing. It performs the complete preprocessing pipeline:
        image loading, resizing, normalization, and tensor conversion.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            dict: Dictionary containing:
                - "images": Preprocessed image tensor of shape (3, height, width)
                - "targets": Target character sequence tensor for CTC loss
        """

        # Loan images from file path
        # Convert it into RGB format as that is the format on which RESNET was trained (3 layers at the start)
        # Plus we might miss some things in grayscale
        image = Image.open(self.image_path[index]).convert("RGB")

        # Get the target using index
        targets = self.target[index]

        # Convert PIL Image to numpy array for albumentations processing
        # Output shape: (height, width, channels)
        # Assume self.resize is None for now
        images_array = np.array(image)

        # Apply the augmentations and extract the image array
        modified_images = self.augmentations(image=images_array)["image"]

        # Current images array are in H,W,C format (HEIGHT, WIDTH, CHANNELS)
        # But pytorch conv layers require the images in C,H,W format (Channels, Height, Width)
        modified_images = np.transpose(modified_images, (2, 0, 1)).astype(np.float32)

        # model weight are typically float32
        # float32 will have us a bit of time
        # loss functions expect float32
        # Classification targets need long for CrossEntropyLoss
        return {
            "images": torch.tensor(modified_images, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }
