import tiktoken

def demonstrate_bpe_handls_unknown_words():
    """
    Shows how the BPE tokenizer breaks down words not in its vocabulary
    into subword units instead of using an 'unknown' token.
    """
    print("--- Demonstrating Out-of-Vocabulary Words ---\n")

    # --- Example 1: A made-up compound word ---
    text1 = "someunknownPlace"
    print(f"Original text: {text1}")

    ids1 = tokenizer.encode(text1)
    print(f"Token IDS: {ids1}")

    strings1 = tokenizer.decode(ids1)
    print(f"Decoded textL {strings1}")
    print(f"\nNotice how 'someunknownPlace' was broken into smaller, know pieces.\n")

    ## --- Example 2: A completely nonsensical word ---
    text2 = "Ajhayw kdib"
    print(f"Original text: {text2}")
    
    ids2 = tokenizer.encode(text2)
    print(f"Token ids: {ids2}")

    strings2 = tokenizer.decode(ids2)
    print(f"Decoded text: {strings2}")
    print("\nEven 'Ajhayw' is broken down into subwords like 'Aj', 'hay', 'w'.")


if __name__ == "__main__":
    # Load the specific GPT tokenizaer
    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"Tokenizer loaded: {tokenizer.name}")
    print(f"Vocavulary size: {tokenizer.n_vocab}\n")

    demonstrate_bpe_handls_unknown_words()