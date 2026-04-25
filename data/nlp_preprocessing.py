import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        # T = 0# max sentence length
        # N = 0# number of samples per class

        all_words = set()
        for i in range(len(positive)):
            # print(positive[i])
            words = positive[i].split(" ")
            [all_words.add(word) for word in words]
            # print(negative[i])
            words = negative[i].split(" ")
            # print(words)
            [all_words.add(word) for word in words]
        # print()
        sorted_words = sorted(list(all_words))
        word_dict = {word:float(idx+1) for idx, word in enumerate(sorted_words)}
        # print(word_dict)

        tensors = []
        for i in range(len(positive)):
            # print(positive[i])
            words = positive[i].split(" ")
            tensor = torch.tensor([word_dict[word] for word in words])
            tensors.append(tensor)

        for i in range(len(positive)):
            # print(negative[i])
            words = negative[i].split(" ")
            # print(words)
            tensor = torch.tensor([word_dict[word] for word in words])
            tensors.append(tensor)

        return torch.nn.utils.rnn.pad_sequence(tensors, padding_value=0, batch_first=True)

