"""DeepLOB model: CNN + Inception + LSTM on top-N levels x time (Model B).

Based on Zhang et al. (2019) 'DeepLOB: Deep Convolutional Neural Networks for Limit Order Books'.
Adapted for variable time windows and top-N order book levels.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InceptionModule(nn.Module):
    """Inception block capturing multi-scale temporal/spatial order book patterns."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding="same")
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding="same")
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding="same")
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.conv_pool = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = F.leaky_relu(self.conv1(x), 0.01)
        b2 = F.leaky_relu(self.conv3(x), 0.01)
        b3 = F.leaky_relu(self.conv5(x), 0.01)
        b4 = F.leaky_relu(self.conv_pool(self.maxpool(x)), 0.01)
        return torch.cat([b1, b2, b3, b4], dim=1)


class DeepLOB(nn.Module):
    """DeepLOB architecture: Convolutions + Inception + LSTM + Linear Head."""

    def __init__(
        self,
        num_classes: int = 3,
        num_levels: int = 10,
        conv_filters: int = 32,
        lstm_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.conv_filters = conv_filters
        self.lstm_hidden = lstm_hidden

        # 1. Convolutional feature extractor
        self.conv1 = nn.Conv2d(1, conv_filters, kernel_size=(1, 2), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(conv_filters)
        self.conv2 = nn.Conv2d(conv_filters, conv_filters, kernel_size=(3, 1), padding="same")
        self.bn2 = nn.BatchNorm2d(conv_filters)

        self.conv3 = nn.Conv2d(conv_filters, conv_filters, kernel_size=(1, 2), stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(conv_filters)
        self.conv4 = nn.Conv2d(conv_filters, conv_filters, kernel_size=(3, 1), padding="same")
        self.bn4 = nn.BatchNorm2d(conv_filters)

        # 2. Inception Module
        self.inception = InceptionModule(conv_filters, conv_filters // 2)
        # Inception output channels = (conv_filters // 2) * 4 = 2 * conv_filters
        incept_channels = (conv_filters // 2) * 4

        # Adaptive pool over order book features axis to fixed dimension 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # 3. LSTM Temporal Dynamics
        self.lstm = nn.LSTM(
            input_size=incept_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # 4. Classification Head
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: Tensor of shape (batch, 1, 4 * num_levels, T) or (batch, 4 *
        num_levels, T).

        Returns:
            Logits of shape (batch, num_classes).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 1. Convolutions
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.01)

        out = F.leaky_relu(self.bn3(self.conv3(out)), 0.01)
        out = F.leaky_relu(self.bn4(self.conv4(out)), 0.01)

        # 2. Inception
        out = self.inception(out)

        # 3. Pool over book levels: (batch, C_incept, 1, T)
        out = self.adaptive_pool(out)
        out = out.squeeze(2)  # (batch, C_incept, T)
        out = out.permute(0, 2, 1)  # (batch, T, C_incept)

        # 4. LSTM
        lstm_out, _ = self.lstm(out)
        # Take the representation from the last time step
        last_hidden = lstm_out[:, -1, :]  # (batch, lstm_hidden)

        # 5. Classifier head
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits
