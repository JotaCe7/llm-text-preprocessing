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

class SimpleTokenizerV2:
    """
    An improved tokenizer that handles out-of-vocabulary words using an <|unk|> token.
    
    Inherits from SimpleTokenizerV1 and overrides the encode method.
    """

    def __init__(self, vocab):
        self.encoder = vocab
        self.decoder = {id:token for token, id in vocab.items()}

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
        text = re.sub(r'\s+([,.;:?_!"()\'])', r'\1', text)
        return text