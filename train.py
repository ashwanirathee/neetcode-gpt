import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss = None
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            indexes = torch.randint(0, len(data)-context_length, size=(batch_size,))
            x = torch.stack([data[i:i + context_length] for i in indexes])
            y = torch.stack([data[i + 1:i + 1 + context_length] for i in indexes])

            logits = model.forward(x)
            print(logits.shape)
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = y.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return round(loss.item(), 4)