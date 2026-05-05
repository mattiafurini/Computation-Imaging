"""
UNet Architecture for End-to-End Super-Resolution.

A classic encoder-decoder CNN with skip connections that learns
the direct mapping from bicubic-upsampled degraded images to
high-resolution reconstructions.

Architecture:
    Input (3, 256, 256) -> Encoder [64->128->256->512] -> Bottleneck [1024]
    -> Decoder [512->256->128->64] -> Output (3, 256, 256)

Each encoder/decoder block uses:
    - 2x Conv3x3 + BatchNorm + LeakyReLU
    - MaxPool2d (encoder) / ConvTranspose2d (decoder)
    - Skip connections via concatenation
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive (Conv3x3 -> BatchNorm -> LeakyReLU) blocks.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    UNet for image-to-image super-resolution.

    Takes a bicubic-upsampled degraded image (3, 256, 256) and outputs
    a reconstructed image (3, 256, 256).

    Parameters
    ----------
    in_channels : int
        Number of input channels (default: 3 for RGB).
    out_channels : int
        Number of output channels (default: 3 for RGB).
    features : list of int
        Number of features at each encoder level.
        Default: [64, 128, 256, 512] for 4-level UNet.
    """

    def __init__(self, in_channels=3, out_channels=3, features=None):
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path
        prev_channels = in_channels
        for feat in features:
            self.encoder_blocks.append(DoubleConv(prev_channels, feat))
            prev_channels = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2  # bottleneck output

        for feat in reversed_features:
            self.upconvs.append(
                nn.ConvTranspose2d(prev_channels, feat, kernel_size=2, stride=2)
            )
            # After concatenation with skip connection: feat (from upconv) + feat (from encoder) = 2*feat
            self.decoder_blocks.append(DoubleConv(feat * 2, feat))
            prev_channels = feat

        # Final 1x1 convolution to map to output channels
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, 3, 256, 256), bicubic-upsampled
            degraded image in range [0, 1].

        Returns
        -------
        torch.Tensor
            Reconstructed image of shape (B, 3, 256, 256) in range [0, 1].
        """
        # Store encoder outputs for skip connections
        skip_connections = []

        # Encoder
        for enc_block in self.encoder_blocks:
            x = enc_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (reverse order of skip connections)
        skip_connections = skip_connections[::-1]

        for i, (upconv, dec_block) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)

            skip = skip_connections[i]

            # Handle potential size mismatch (e.g., odd dimensions)
            if x.shape != skip.shape:
                x = nn.functional.pad(x, [
                    0, skip.shape[3] - x.shape[3],
                    0, skip.shape[2] - x.shape[2]
                ])

            # Concatenate skip connection
            x = torch.cat([skip, x], dim=1)
            x = dec_block(x)

        # Final convolution + sigmoid to ensure [0, 1] output
        x = self.final_conv(x)
        x = torch.sigmoid(x)

        return x


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test: verify architecture shapes
    model = UNet(in_channels=3, out_channels=3)
    print(f"UNet parameters: {count_parameters(model):,}")

    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    assert output.shape == dummy_input.shape, "Output shape mismatch!"
    print("✓ Architecture test passed!")
