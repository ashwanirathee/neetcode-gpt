import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.first_layer = nn.Linear(784, 512)
        self.relu_layer = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=0.2)
        self.second_layer = nn.Linear(512,10)
        self.sigmoid_layer = nn.Sigmoid()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512,10),
            nn.Sigmoid()
        )

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x = self.first_layer(images)
        x = self.relu_layer(x)
        x = self.dropout_layer(x)
        x = self.second_layer(x)
        x = self.sigmoid_layer(x)
        return x