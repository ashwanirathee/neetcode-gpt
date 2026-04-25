import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        print("Number of tokens in a sentence:", seq_len)
        print("Size of each token embedding vector:", d_model)
        positions = np.arange(seq_len).reshape(-1, 1) # number of max tokens in a sentence

        PE_res = np.zeros((seq_len, d_model)) # the resultant matrix for each sentence, is for each position, we have it's encoding in terms of embedding size

        even_i = np.arange(0, d_model, 2) # to handle even case
        odd_i = np.arange(1, d_model, 2) # to handle odd casees

        PE_res[:, 0::2] = np.sin(positions / (10000 ** (even_i / d_model)))
        PE_res[:, 1::2] = np.cos(positions / (10000 ** ((odd_i - 1) / d_model)))

        return np.round(PE_res, 5)