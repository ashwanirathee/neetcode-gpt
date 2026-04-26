import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        tokenized_data = raw_dataset.split()
        starts = torch.randint(0, len(tokenized_data)-context_length, size=(batch_size,))
        # print(starts)

        X = [[""] * context_length for j in range(batch_size)]
        Y = [[""] * context_length for j in range(batch_size)]
        for row, i in enumerate(starts):
            # print(row,i)
            i = int(i.item())
            X[row] = tokenized_data[i:i+context_length]
            Y[row] = tokenized_data[i+1:i+context_length+1]

        return (X,Y)
