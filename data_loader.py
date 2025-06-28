"""
Contains the GPTDatasetV1 class and a utility function to create a PyTorch
DataLoader for training a GPT-style model.
"""

import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        """
        Args:
            txt (str): The full raw text to process.
            tokenizer: The tokenizer instance (e.g. from tiktoken).
            max_length (int): The maximum length of each input sequence (context size).
            stride (int): The step size to move the sliding window across the text.git status
        """
        self.input_ids = []
        self.target_ids = []

        # Step 1: Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # Step 2: Use a sliding window to create chunks of text
        for i in range(0,len(token_ids), stride):
            # The input chunk is a sequence of tokens of size max_length
            input_chunk = token_ids[i:i + max_length]
            # The target chunk is the same sequence, shifted by one token to the right
            target_chunk = token_ids[i + 1: i + max_length + 1]

            # Convert chunks to PyTorch tensors and store them
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self) -> int:
        """Returns the total number of chunks in the dataset."""
        return len(self.input_ids)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single input-target pair from the dataset at a given index.
        
        Args:
            index (int): The index of sample to retrieve.
        
        Returns:
            A tuple containing the input tensor and the target tensor.
        """
        return self.input_ids[index], self.target_ids[index]
