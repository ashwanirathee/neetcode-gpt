from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        stoi = {}
        max_val = 0
        data = sorted(list(set(list(text))))
        for i in data:
            stoi[i] = max_val
            max_val = max_val + 1
        # print(stoi)
        itos = {}
        for key, value in stoi.items():
            itos[value] = key
        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        encoding = []
        for i in text:
            encoding.append(stoi[i])
        return encoding

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decoding = []
        for i in ids:
            decoding.append(itos[i])
        return "".join(decoding)
