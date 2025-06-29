import re

# --- Version 1: The Basic Tokenizer ---

class SimpleTokenizerV1:
    """A simple word-based tokenizer that maps tokens to integer IDs."""

    def __init__(self, vocab):
        """
        Initializes the tokenizer with a given vocabulary.

        Args:
            vocab (dict): A dictionary mapping string tokens to integer IDs.
        """
        self.encoder = vocab
        self.decoder = {id:token for token, id in vocab.items()}

    def encode(self, text):
        """
        Converts a string of text into a list of token IDs.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[int]: A list of integer token IDs.
        
        Raises:
            KeyError: If a token in the text is not found in the vocabulary.
        """
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.encoder[item] for item in preprocessed]
        return ids

    def decode(self, ids):
        """
        Converts a list of token IDs back into a string of text.

        Args:
            ids (list[int]): The list of token IDs to decode.

        Returns:
            str: The reconstructed text.
        """
        text = " ".join([self.decoder[id] for id in ids])
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text
    

# --- Version 2: Handling Unknown Words ---

class SimpleTokenizerV2(SimpleTokenizerV1):
    """
    An improved tokenizer that handles out-of-vocabulary words using an <|unk|> token.
    
    Inherits from SimpleTokenizerV1 and overrides the encode method.
    """

    def encode(self, text):
        """
        Converts a string of text into a list of token IDs.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[int]: A list of integer token IDs.
        """
        preprocessed = re.split(r'([,.;:?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed_with_unk  = [
            item if item in self.encoder
            else "<|unk|>" for item in preprocessed
        ]
        ids = [self.encoder[token] for token in preprocessed_with_unk ]
        return ids

# --- Main block to make the script runnable for demonstration ---
if __name__ == "__main__":
    
    # 1. Create a sample vocabulary from a small text
    sample_text = "Hello, world. This is a test."
    split_tokens = re.split(r'([,.:;?_!"()\']|--|\s)', sample_text)
    cleaned_tokens = [item.strip() for item in split_tokens if item.strip()]
    all_unique_tokens = sorted(list(set(cleaned_tokens)))
    base_vocab = {token: i for i, token in enumerate(all_unique_tokens)}
    
    # 2. Demonstrate SimpleTokenizerV1 (which will fail on unknown words)
    print("--- Demonstrating SimpleTokenizerV1 (will fail) ---")
    tokenizer_v1 = SimpleTokenizerV1(base_vocab)
    text_with_unknown = "Hello, is your world only a test?"
    print(f"Original Text: '{text_with_unknown}'")
    print(f"Base Vocabulary: {list(base_vocab.keys())}")
    try:
        tokenizer_v1.encode(text_with_unknown)
    except KeyError as e:
        print(f"\nSUCCESSFULLY FAILED: SimpleTokenizerV1 failed as expected because {e} is not in the vocabulary.\n")

    # 3. Demonstrate SimpleTokenizerV2 (which will succeed)
    print("--- Demonstrating SimpleTokenizerV2 (will succeed) ---")
    # For V2, we add the special '<|unk|>' token to the vocabulary
    v2_vocab = base_vocab.copy()
    v2_vocab["<|unk|>"] = len(v2_vocab)
    tokenizer_v2 = SimpleTokenizerV2(v2_vocab)
    
    encoded_v2 = tokenizer_v2.encode(text_with_unknown)
    print(f"Encoded IDs with V2: {encoded_v2}")
    
    decoded_v2 = tokenizer_v2.decode(encoded_v2)
    print(f"Decoded Text with V2: '{decoded_v2}'")
    print("\nSUCCESS: SimpleTokenizerV2 handled the unknown words using the '<|unk|>' token.")