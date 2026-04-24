import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        res = []
        with torch.no_grad():
            for module in model.children():
                print(module)
                x = module(x)
                if isinstance(module, nn.ReLU):
                    re = (x==0).all(dim=0).float().mean().item()
                    res.append(round(re, 4))
        return res


    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        prev = dead_fractions[0]
        strictly_increasing = True
        max_dead_frac = max(dead_fractions)
        for idx, dead_frac in enumerate(dead_fractions):
            if dead_frac > 0.5:
                return'use_leaky_relu'
            elif idx ==0 and dead_frac > 0.3:
                return'reinitialize'
            elif idx > 0 and strictly_increasing == True and dead_frac < prev:
                strictly_increasing = False
            prev = dead_frac

        if strictly_increasing == True and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'
        if max_dead_frac < 0.1:
            return 'healthy'
        
        return 'healthy'

        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise 
