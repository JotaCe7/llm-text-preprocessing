# A Deep Dive into LLM Text Preprocessing

This repository provides a hands-on guide to the essential text preprocessing pipeline required for building Large Language Models (LLMs). It breaks down the journey from raw text to model-ready tensors through modular, clear, commented Python code.

## The Importance of Preprocessing

Before a Large Language Model can learn from text, the raw characters must be converted into a numerical format it can understand. This multi-stage process, known as preprocessing, is the foundational first step in any NLP project. This repository explores the key stages of this pipeline.

---

## Project Structure
```
llm-text-preprocessing/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   └── the-verdict.txt
├── notebooks/
│   └── text_preprocessing_walkthrough.ipynb
├── custom_tokenizer.py   # Demonstrates building a simple tokenizer
├── bpe_demo.py           # Demonstrates the modern BPE tokenizer
├── data_loader.py        # Contains the PyTorch DataLoader logic
└── main_pipeline.py      # Runs the full pipeline from text to embeddings
```

## Core Concepts Demonstrated

This project is organized to demonstrate the critical stages of turning text into a format suitable for a model like GPT.

1.  **Building a Simple Tokenizer (`custom_tokenizer.py`)**
    * This script shows the step-by-step evolution of a custom tokenizer. It starts with a basic version (`V1`) that fails on unknown words, then introduces an improved version (`V2`) that solves this problem using an `<|unk|>` token.

2.  **Advanced Tokenization with BPE (`bpe_demo.py`)**
    * A focused demonstration of the powerful Byte-Pair Encoding (BPE) algorithm using the `tiktoken` library. It shows how modern tokenizers handle any word by breaking it into known subword units, eliminating the need for an "unknown" token.

3.  **Preparing Data for Training (`data_loader.py`)**
    * This module contains the logic to structure tokenized text into `(input, target)` pairs and create an efficient PyTorch `DataLoader` for model training.

4.  **The Full End-to-End Pipeline (`main_pipeline.py`)**
    * This is the main script that ties everything together. It loads raw text, utilizes the tokenizers, creates data batches via the `DataLoader`, and prepares the final token and positional embeddings required by a model.

---
## How to Run

1.  **Clone the Repository:**
    You can find the repository here: [https://github.com/JotaCe7/llm-text-preprocessing.git](https://github.com/JotaCe7/llm-text-preprocessing.git)

    Use the following command to clone it to your local machine:
    ```bash
    git clone https://github.com/JotaCe7/llm-text-preprocessing.git
    cd llm-text-preprocessing
    ```

2.  **Set Up Environment and Install Dependencies:**
    ```bash
    python -m venv env
    .\env\Scripts\activate
    pip install -r requirements.txt
    ```
    

3.  **Run the Main Pipeline:**
    To see all the steps in action, run the main script.
    ```bash
    python main_pipeline.py
    ```

4.  **Run Focused Demos:**
    To explore specific concepts, run the individual demo scripts.
    ```bash
    # See the evolution from a basic tokenizer to one that handles unknown words
    python custom_tokenizer.py

    # See how the modern BPE tokenizer works
    python bpe_demo.py
    ```