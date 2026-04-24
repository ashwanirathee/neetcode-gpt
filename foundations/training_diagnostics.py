import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        result = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    mean_val = round(x.mean().item(), 4)
                    std_val = round(x.std().item(), 4)
                    if x.dim() >= 2:
                        dead_frac = round(((x <= 0).all(dim=0)).float().mean().item(), 4)
                    else:
                        dead_frac = round((x <= 0).float().mean().item(), 4)
                    result.append({'mean': mean_val, 'std': std_val, 'dead_fraction': dead_frac})
        return result

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        result = []
        for module in model.children():
            if isinstance(module, nn.Linear):
                x = module.weight.grad
                mean_val = round(x.mean().item(), 4)
                std_val = round(x.std().item(), 4)
                dead_frac = torch.norm(x).item()
                result.append({'mean': mean_val, 'std': std_val, 'norm': dead_frac})
        return result

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)

        dead_fraction_case = False
        exploding_gradients_case = False
        vanishing_gradients_case = False

        for i in activation_stats:
            print(i)
            if i['dead_fraction'] > 0.5:
                dead_fraction_case = True
            if i['std'] > 10.0:
                exploding_gradients_case = True
            if i['std'] < 0.1:
                vanishing_gradients_case = True

        length = len(gradient_stats)
        for idx, j in enumerate(gradient_stats):
            if j['norm'] > 1000:
                exploding_gradients_case = True
            if idx == length-1 and j['norm'] < 1e-5:
                vanishing_gradients_case = True

        if dead_fraction_case == True:
            return 'dead_neurons'
        elif exploding_gradients_case == True:
            return 'exploding_gradients'
        elif vanishing_gradients_case == True:
            return 'vanishing_gradients'
        else:
            return 'healthy'
