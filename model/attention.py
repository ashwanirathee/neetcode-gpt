import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.w_key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.w_query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.w_value = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.attention_dim = attention_dim
        self.softmax = nn.Softmax(dim=2)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        k = self.w_key(embedded)
        q = self.w_query(embedded)
        v = self.w_value(embedded)

        attention_scores = q @ k.transpose(-1,1) / (self.attention_dim ** 0.5)

        seq_len = embedded.shape[1] # length of each sentence vecotr
        mask = torch.tril(torch.ones(seq_len, seq_len, device=embedded.device))
        # this above create causal mask, which means keeps attention only in forward
        # direction, and avoid backword correlation
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        # this above applies that
        softmax_output = self.softmax(attention_scores)
        return torch.round(softmax_output @ v, decimals=4)
