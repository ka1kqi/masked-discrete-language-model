"""Tokenizer wrapper for the diffusion LM."""

from transformers import AutoTokenizer


def get_tokenizer(name: str = "bert-base-uncased") -> AutoTokenizer:
    """Load a pretrained tokenizer.

    We use BERT's tokenizer for its [MASK] token support and
    reasonable 30k vocab size.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    return tokenizer


def tokenize_dataset(dataset, tokenizer, max_seq_len: int = 128, text_column: str = "text"):
    """Tokenize a HuggingFace dataset, chunking into fixed-length sequences.

    Args:
        dataset: HuggingFace Dataset
        tokenizer: tokenizer instance
        max_seq_len: fixed sequence length
        text_column: name of the text column

    Returns:
        Tokenized dataset with 'input_ids' column of shape (seq_len,)
    """

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len,
            return_special_tokens_mask=True,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=dataset.column_names,
    )
    tokenized.set_format("torch", columns=["input_ids", "attention_mask"])
    return tokenized
