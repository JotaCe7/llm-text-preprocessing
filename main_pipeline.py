import re
import torch
import tiktoken

from custom_tokenizer import SimpleTokenizerV2
from data_loader import create_dataloader

def demonstrate_data_loading(filepath: str) -> str:
    """Loads raw text from a file and prints basic statistics."""
    print("--- 1. Loading Data ---")
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"Successfully loaded {filepath}")
    print(f"Total characters: {len(raw_text)}")
    print(f"First 100 characters: {raw_text[:100]}\n")
    return raw_text

def demonstrate_custom_tokenizer(raw_text: str):
    """Demonstrates the custom V2 tokenizer with <unk> token handling."""
    print("--- 2. Demonstrating Simple Custom Tokenizer (V2) ---")
    
    # Build a vocabulary from the raw text using our simple regex
    split_tokens = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    cleaned_tokens = [item.strip() for item in split_tokens if item.strip()]
    all_unique_tokens = sorted(list(set(cleaned_tokens)))
    vocab = {token: i for i, token in enumerate(all_unique_tokens)}
    vocab["<|unk|>"] = len(vocab)

    # Instantiate and test the tokenizer on a sentence with unknown words
    custom_tokenizer = SimpleTokenizerV2(vocab)
    text_to_test = "Hello, do you like tea?"
    encoded = custom_tokenizer.encode(text_to_test)
    decoded = custom_tokenizer.decode(encoded)
    
    print(f"Original: '{text_to_test}'")
    print(f"Encoded (V2): {encoded}")
    print(f"Decoded (V2): '{decoded}'\n")

def demonstrate_bpe_tokenizer(raw_text: str) -> tiktoken.Encoding:
    """Demonstrates the tiktoken BPE tokenizer."""
    print("--- 3. Demonstrating BPE Tokenizer (tiktoken) ---")
    tokenizer = tiktoken.get_encoding("gpt2")
    bpe_encoded = tokenizer.encode(raw_text)
    print(f"Text tokenized with BPE. Total tokens: {len(bpe_encoded)}\n")
    return tokenizer

def demonstrate_dataloader(raw_text: str, max_length: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Demonstrates the PyTorch DataLoader."""
    print("--- 4. Demonstrating DataLoader for Training ---")
    dataloader = create_dataloader(
        txt=raw_text, batch_size=batch_size, max_length=max_length,
        stride=max_length, shuffle=False
    )
    inputs, targets = next(iter(dataloader))
    
    print("Sample Batch from DataLoader:")
    print(f"Inputs Shape: {inputs.shape}")
    print(f"Targets Shape: {targets.shape}")
    print("Inputs:\n", inputs)
    print("Targets:\n", targets)
    print("\n")
    return inputs, targets

def demonstrate_embeddings(inputs: torch.Tensor, tokenizer, max_length: int):
    """Demonstrates token and positional embeddings."""
    print("--- 5. Demonstrating Token and Positional Embeddings ---")
    
    vocab_size = tokenizer.n_vocab 
    embedding_dim = 256

    token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    pos_embedding_layer = torch.nn.Embedding(max_length, embedding_dim)

    token_embeddings = token_embedding_layer(inputs)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    # PyTorch broadcasts pos_embeddings across the batch dimension for the addition
    input_embeddings = token_embeddings + pos_embeddings
    
    print(f"Input Token IDs shape: {inputs.shape}")
    print(f"Token Embeddings shape: {token_embeddings.shape}")
    print(f"Positional Embeddings shape: {pos_embeddings.shape}")
    print(f"Final Input Embeddings for model: {input_embeddings.shape}")

def main():
    """Runs the full preprocessing pipeline demonstration."""
    
    # --- Configuration ---
    FILE_PATH = "data/the-verdict.txt"
    CONTEXT_LENGTH = 4
    BATCH_SIZE = 8

    # --- Run Pipeline Steps ---
    raw_text = demonstrate_data_loading(FILE_PATH)
    demonstrate_custom_tokenizer(raw_text)
    bpe_tokenizer = demonstrate_bpe_tokenizer(raw_text)
    inputs, _ = demonstrate_dataloader(raw_text, max_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE)
    demonstrate_embeddings(inputs, tokenizer=bpe_tokenizer, max_length=CONTEXT_LENGTH)

if __name__ == "__main__":
    main()