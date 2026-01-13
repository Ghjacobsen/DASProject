import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for DAS Anomaly Detection.
    
    Structure:
    - Encoder: Compresses 2D acoustic patches into a latent vector.
    - Bottleneck: Forces the model to learn only the most persistent features (noise).
    - Decoder: Reconstructs the patch from the latent vector.
    """

    def __init__(self, config, latent_dim):
        super(ConvAutoencoder, self).__init__()
        
        self.patch_size = config['data']['patch_size']
        
        # --- Encoder ---
        self.encoder = nn.Sequential(
            # Input: (1, 128, 128)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # -> (16, 64, 64)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 16, 16)
            nn.ReLU()
        )
        
        # Calculate flatten size automatically based on patch size logic above
        # For 128x128 input reduced by factor of 8 (2^3 layers) -> 16x16
        self.flatten_dim = 64 * (self.patch_size[0] // 8) * (self.patch_size[1] // 8)
        
        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.flatten_dim),
            nn.ReLU()
        )
        
        # --- Decoder ---
        self.decoder_input = nn.Unflatten(1, (64, self.patch_size[0] // 8, self.patch_size[1] // 8))
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # No final activation (linear) because inputs are Z-scored (can be negative)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder_input(x)
        x = self.decoder(x)
        return x