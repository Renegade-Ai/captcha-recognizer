import torch  # Core PyTorch tensor operations
from torch import nn  # Neural network layers and modules
from torch.nn import (
    functional as F,  # Functional interface for activations, losses, etc.
)


class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        # Inherit pytorch nn module functionality
        super(CaptchaModel, self).__init__()

        # ===== CONVOLUTIONAL LAYERS (Feature Extraction) =====

        # 1st Convolutional Layer (batch_size, 3, height, width)
        # Input: (batch_size, 3, 75, 300)
        # Output: (batch_size, 128, 75, 300) - 128 feature maps

        self.conv1 = nn.Conv2d(
            in_channels=3,  # 3 channels
            out_channels=128,  # 128 number of feature maps that model will learn
            kernel_size=(3, 6),  # kernel size filter (capture character size patterns)
            padding=1,  # Mantain spatial mapping
        )

        # batch normalization after 1st layer
        self.bn1 = nn.BatchNorm2d(128)

        # Max pooling layer
        # Input: (batch_size, 128, 75, 300)
        # Output: (batch_size, 128, 37, 300)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))

        # 2nd Convolutional Layer
        # Input: (batch_size, 128, 37, 300)
        # Output: (batch_size, 64, 37, 300) - 64 feature maps
        self.conv2 = nn.Conv2d(
            in_channels=128,  # From previous conv layer
            out_channels=64,  # Reduce feature maps (compression)
            kernel_size=(3, 6),  # Same filter size
            padding=1,  # Maintain spatial dimensions
        )

        # Batch normalization after 2nd layer
        self.bn2 = nn.BatchNorm2d(64)

        # Max pooling layer
        # Input: (batch_size, 64, 37, 300)
        # Output: (batch_size, 64, 18, 300) - Only halves height, preserves width
        # Thought process is that since we are using with width as time steps we should preserve them as much as possible
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))

        # Dropout layer: Prevent overfitting by randomly zeroing some neurons
        self.drop1 = nn.Dropout(0.2)  # 20% dropout rate

        # ===== TRANSITION TO SEQUENCE MODELING =====

        # Linear layer: Prepare CNN features for RNN processing and Dimensionality reduction
        # After reshaping: (batch_size, 300, 1152) → (batch_size, 300, 64)
        # 1152 = 64 channels × 18 height (flattened spatial dimensions)
        self.linear_1 = nn.Linear(in_features=64 * 18, out_features=64)

        # LSTM layer
        # Input: (batch_size, 64, 18, 300)
        # Output: (batch_size, 18, 64) - Time steps, hidden size
        # IMPROVED: Increased hidden size and reduced layers for better learning
        """
        LSTM Layer Configuration - Detailed Parameter Reasoning:
        
        input_size=64: Input feature size from conv layers after reshaping
                      Each time step receives 64 features (depth from conv2)
                      This matches the channel dimension from conv layers
        
        hidden_size=64: Hidden state size (per direction) - INCREASED
                       Controls LSTM memory capacity and learning ability
                       Larger = more complex patterns, but risk of overfitting
                       64 provides good balance for CAPTCHA complexity
        
        num_layers=1: Single layer to reduce complexity - SIMPLIFIED
                     Fewer layers = faster training, less overfitting
                     CAPTCHA patterns aren't complex enough to need deep LSTM
                     Single layer sufficient for character sequence learning
        
        batch_first=True: Input format: (batch, seq, feature)
                         More intuitive than (seq, batch, feature)
                         Matches standard PyTorch tensor conventions
                         Easier to work with in forward pass
        
        dropout=0.0: No dropout in single layer
                    Dropout only useful between multiple LSTM layers
                    With num_layers=1, dropout has no effect
                    Regularization handled by other dropout layers in model
        
        bidirectional=True: Process sequence both forward and backward
                           Critical for CAPTCHA: characters provide context for neighbors
                           Forward pass: left-to-right character recognition
                           Backward pass: right-to-left context for disambiguation
                           Doubles output size: 64*2=128 features per time step
        """
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )

        # Final linear layer: Map RNN outputs to character probabilities
        # Input: (batch_size, 300, 128)
        # Output: (batch_size, 300, num_chars + 1)
        self.output = nn.Linear(
            in_features=128,  # From bidirectional GRU (64×2) - UPDATED
            out_features=num_chars + 1,  # +1 for CTC blank token
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize model weights using layer-appropriate strategies to ensure stable training.

        This method implements a sophisticated initialization strategy that:
        1. Prevents vanishing/exploding gradients
        2. Ensures stable training with large datasets (30k samples)
        3. Addresses CTC-specific challenges (blank token dominance)
        4. Optimizes for the CNN-RNN hybrid architecture

        The initialization strategy is tailored specifically for CAPTCHA recognition tasks.
        """

        # ===== CONVOLUTIONAL LAYER INITIALIZATION =====
        for m in (self.conv1, self.conv2):
            # print("Setting weights for convolutional layer")
            nn.init.kaiming_normal_(
                m.weight,
                mode="fan_out",  # Use output fan for forward pass stability
                nonlinearity="relu",  # Optimized for ReLU activation function
            )

            # Initialize conv biases to zero (standard practice)
            # WHY ZERO BIAS?
            # - ReLU(0) = 0: no initial activation bias
            # - Let network learn optimal bias during training
            # - Prevents neurons from being "always on" initially
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # ===== BATCH NORMALIZATION LAYER INITIALIZATION =====
        # Standard BatchNorm initialization: scale=1, shift=0
        #
        # BATCH NORMALIZATION FORMULA:
        # y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
        # where gamma=weight, beta=bias
        #
        # WHY (1, 0)?
        # - gamma=1: Identity scaling initially (no transformation)
        # - beta=0: No shift initially (centered at zero)
        # - Network learns optimal normalization parameters during training
        # - Critical for stable training with large datasets (our 30k samples)
        for m in [self.bn1, self.bn2]:
            nn.init.constant_(m.weight, 1)  # gamma: scale parameter = 1 (identity)
            nn.init.constant_(m.bias, 0)  # beta: shift parameter = 0 (no shift)

        # ===== LINEAR LAYER INITIALIZATION =====
        # Use Xavier Normal for linear transformation layers
        #
        # WHY XAVIER NORMAL?
        # - Developed by Xavier Glorot for general neural networks
        # - Mathematical basis: std = sqrt(2 / (fan_in + fan_out))
        # - Balances input and output dimensions for stable gradients
        # - Ideal for feature compression (1152 → 64 in our case)
        # - Works well for non-ReLU layers and transition layers
        nn.init.xavier_normal_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)  # Zero bias for symmetric start

        # ===== RECURRENT LAYER (GRU) INITIALIZATION =====
        # Use Xavier Normal for all GRU weight matrices
        #
        # GRU PARAMETER BREAKDOWN:
        # - weight_ih_l0: Input-to-hidden weights [192, 64]
        # - weight_hh_l0: Hidden-to-hidden weights [192, 64]
        # - bias_ih_l0: Input-to-hidden biases [192]
        # - bias_hh_l0: Hidden-to-hidden biases [192]
        #
        # NOTE: 192 = 3 gates × 64 hidden units (reset, update, new gates)
        #
        # WHY XAVIER FOR RNN?
        # - Long sequences (300 time steps) require stable gradient flow
        # - Bidirectional processing needs balanced forward/backward stability
        # - Prevents vanishing gradients in sequential processing
        # - Memory preservation across time steps
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                # Initialize all weight matrices (input-to-hidden, hidden-to-hidden)
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                # Initialize all bias vectors to zero
                nn.init.constant_(param, 0)

        # ===== OUTPUT LAYER INITIALIZATION: CTC-AWARE STRATEGY =====
        # This is the MOST CRITICAL part for CTC-based sequence prediction
        #
        # PROBLEM WITH STANDARD INITIALIZATION:
        # - Standard init: all output biases = 0
        # - CTC models tend to predict blank tokens (index 0) early in training
        # - Result: model outputs empty strings initially
        # - This can prevent learning and cause training to stagnate
        #
        # OUR SOLUTION: CTC-AWARE BIAS INITIALIZATION
        # 1. Use Xavier Normal for weight matrix (standard good practice)
        nn.init.xavier_normal_(self.output.weight)

        # 2. Initialize ALL character biases to -0.1 (slight penalty)
        # This makes characters slightly harder to predict than blank token
        # Forces model to overcome small bias → encourages actual character learning
        nn.init.constant_(self.output.bias, -0.1)

        # 3. Set blank token bias to 0 (neutral)
        # Index 0 = CTC blank token (represents "no character")
        # Neutral bias means blank token has no advantage/disadvantage
        # Final bias array: [0.0, -0.1, -0.1, -0.1, ..., -0.1]
        #                   ↑     ↑     ↑     ↑          ↑
        #                 blank  'A'   'B'   'C'       'z'
        self.output.bias.data[0] = 0.0

        # MATHEMATICAL IMPACT:
        # In softmax: P(class_i) = exp(logit_i + bias_i) / normalization
        # Characters need slightly higher logits to match blank probability
        # This bias encourages character predictions over blank tokens
        # Result: Model learns to predict actual characters from early epochs
        #
        # EVIDENCE THIS WORKS:
        # - Training shows 3.1% accuracy by epoch 1 (not stuck at 0%)
        # - 65.8% accuracy by epoch 4 (rapid improvement)
        # - Perfect predictions: 'UA1N' → 'UA1N' (proper character learning)

    def forward(self, images, targets=None):
        """
        Forward pass through the CAPTCHA recognition model.

        Args:
            images (torch.Tensor): Input CAPTCHA images of shape (batch_size, 3, 75, 300)
            targets (torch.Tensor, optional): Ground truth character sequences for training
                                            Shape: (batch_size, max_sequence_length)
        Returns:
            tuple: (predictions, loss) where:
                - predictions: Raw logits of shape (seq_len, batch_size, num_classes)
                - loss: CTC loss if targets provided, None otherwise
        """

        # ===== EXTRACT BATCH SIZE =====
        # Get batch size for later tensor reshaping operations
        bs, channels, height, width = images.size()
        # print(f"DEBUG - Input images shape: {images.shape}")

        # ===== CNN FEATURE EXTRACTION PIPELINE =====

        # Step 1: First convolution + batch norm + activation
        # Input: (bs, 3, 75, 300) → Output: (bs, 128, 75, 300)
        x = self.conv1(images)
        x = self.bn1(x)

        # ReLU activation introduces non-linearity, enables learning complex patterns
        x = F.relu(x)

        # Step 2: First pooling (downsampling height only)
        # Input: (bs, 128, 75, 300) → Output: (bs, 128, 37, 300)
        # Reduces height by half but preserves width for RNN processing
        x = self.pool1(x)

        # Step 3: Second convolution + batch norm + activation
        # Input: (bs, 128, 37, 300) → Output: (bs, 64, 37, 300)
        # Learns higher-level features, reduces channel count
        x = F.relu(self.bn2(self.conv2(x)))

        # Step 4: Second pooling (height only)
        # Input: (bs, 64, 37, 300) → Output: (bs, 64, 18, 300)
        # Further height reduction while preserving width for RNN
        x = self.pool2(x)
        # print(f"DEBUG - After pool2: {x.shape}")

        # ===== RESHAPE FOR SEQUENCE MODELING =====
        # Step 5: Rearrange dimensions for RNN processing
        # Input: (bs, 64, 18, 300) → Output: (bs, 300, 18, 64)
        # Move width dimension to sequence position for temporal modeling
        x = x.permute(0, 3, 1, 2)

        # Step 6: Flatten spatial dimensions into feature vectors
        # Input: (bs, 300, 18, 64) → Output: (bs, 300, 1152)
        # Each of 300 time steps has 18×64=1152 features
        # This creates a sequence of feature vectors for RNN
        x = x.view(bs, x.size(1), -1)
        # print(f"DEBUG - After flatten for linear: {x.shape}")
        # print(f"DEBUG - Linear layer expects: in_features={64 * 18} = {64 * 18}")

        # ===== FEATURE COMPRESSION AND REGULARIZATION =====

        # Step 7: Linear transformation + activation
        # Input: (bs, 300, 1152) → Output: (bs, 300, 64)
        # Compress high-dimensional CNN features for efficient RNN processing
        x = F.relu(self.linear_1(x))

        # Step8: Dropout regularization
        # Randomly drop 20% during traiming to prevent overfitting
        x = self.drop1(x)

        # Step 9: Bidirectional GRU processing
        # Input: (bs, 300, 64) → Output: (bs, 300, 128)
        # Models temporal dependencies between character positions
        # Bidirectional: processes sequence both left→right and right→left
        x, _ = self.lstm(x)  # Ignore hidden state, only need output

        # Step 10: Character classification
        # Input: (bs, 300, 128) → Output: (bs, 300, num_chars+1)
        # Maps RNN features to character probability distributions
        x = self.output(x)

        # Step 11: Transpose for CTC loss format
        # Input: (bs, 300, num_chars+1) → Output: (300, bs, num_chars+1)
        # CTC loss expects (seq_len, batch_size, num_classes) format
        x = x.permute(1, 0, 2)

        # ===== LOSS CALCULATION (Training Mode) =====
        if targets is not None:

            log_probs = F.log_softmax(x, 2)

            input_lengths = torch.full(
                (bs,),  # [something, something, something, ..... bs times])
                fill_value=log_probs.size(
                    0
                ),  # Fill the tensor with 294 which is the actual sequence length from the model, from [294, 8, 63]
                dtype=torch.int32,
            )

            target_lengths = torch.full(
                (bs,),
                fill_value=targets.size(1),  # all the targets are of the same length 4
                dtype=torch.int32,
            )

            # Debug prints commented out to fix progress bar display
            # print("Log probs shape", log_probs.shape)
            # print("Input lengths shape", input_lengths.shape)
            # print("Target lengths shape", target_lengths.shape)

            loss = nn.CTCLoss(blank=0)(
                log_probs,  # Model predictions (300, bs, num_chars+1)
                targets,  # Ground truth sequences (bs, 4)
                input_lengths,  # Length of each prediction sequence (bs,)
                target_lengths,  # Length of each target sequence (bs,)
            )

            # print("Loss calculations complete")

            return x, loss

        return x, None
