from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer
from typing import List, Tuple
from torch.utils.data import DataLoader


def get_dataloaders_alpaca(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    eval_split: float = 0.1,
    batch_size: int = 4,
    max_length: int = None,
) -> Tuple[DataLoader, DataLoader]:
    """Given the Alpaca dataset (e.g., 'tatsu-lab/alpaca'), create
    train and val dataloaders. It provides the possibility to filter by `max_lenght`
    instructions. Dataloaders have the following keys:
    'input_ids': encode text.
    'attention_mask': padding attention mask
    'labels': same as input_ids, but instruction tokens are replaced by tokenizer.pad_token_id.

    Args:
        dataset (DatasetDict): input dataset.
        tokenizer (AutoTokenizer): input tokenizer.
        eval_split (float, optional): eval split. Defaults to 0.1.
        batch_size (int, optional): batch size. Defaults to 4.
        max_length (int, optional): filter max lenght. Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader]: train and val dataloaders.
    """
    # Create evaluation set
    dataset = dataset["train"].train_test_split(test_size=eval_split)

    # Add EOS
    def add_EOS_token(example):
        example["text"] = example["text"] + tokenizer.eos_token
        return example

    dataset = dataset.map(add_EOS_token)

    def tokenize_function_text(examples):
        # max_length=None => use the model max length
        output = {}
        output["input_ids"] = tokenizer(
            examples["text"],
            truncation=False,
            max_length=None,
            add_special_tokens=True,
        )["input_ids"]
        return output

    def tokenize_function_answer(examples):
        # max_length=None => use the model max length (it's actually the default)
        output = {}
        output["answer"] = tokenizer(
            examples["output"],
            truncation=True,
            max_length=None,
            add_special_tokens=True,
        )["input_ids"]
        return output

    tokenized_datasets = dataset.map(
        tokenize_function_text,
        batched=True,
        remove_columns=["instruction", "input", "text"],
    )

    tokenized_datasets = tokenized_datasets.map(
        tokenize_function_answer, batched=True, remove_columns=["output"]
    )

    # Filter data
    if max_length:

        def fiter_by_max_lenght(example) -> str:
            return len(example["input_ids"]) <= max_length

        tokenized_datasets = DatasetDict(
            {
                split: tokenized_datasets[split].filter(fiter_by_max_lenght)
                for split in tokenized_datasets.keys()
            }
        )

    print(
        f"Train: {tokenized_datasets['train'].num_rows} | Eval: {tokenized_datasets['test'].num_rows}"
    )

    def collate_fn(examples):
        for example in examples:
            example["labels"] = example["input_ids"].copy()

            answer_len = len(example["answer"])

            # Mask instruction with pad_token_id
            example["labels"][: len(example["input_ids"]) - answer_len] = [
                tokenizer.pad_token_id
            ] * (len(example["input_ids"]) - answer_len)

        input_ids_n_attention = tokenizer.pad(
            {"input_ids": [example["input_ids"] for example in examples]},
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        labels = tokenizer.pad(
            {"input_ids": [example["labels"] for example in examples]},
            padding="longest",
            max_length=max_length,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        return dict(
            input_ids=input_ids_n_attention["input_ids"],
            attention_mask=input_ids_n_attention["attention_mask"],
            labels=labels["input_ids"],
        )

    # Instantiate dataloaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        pin_memory=True,  # Data pre-loading
        num_workers=4,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        tokenized_datasets["test"],
        shuffle=True,
        pin_memory=True,  # Data pre-loading
        num_workers=4,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=True,
    )

    return train_dataloader, val_dataloader


def compute_max_avg(inputs: List[List[int]]) -> Tuple[int, float]:
    max_value = 0
    avg = []
    for sample in inputs:
        sample_len = len(sample)
        avg.append(sample_len)

        if sample_len > max_value:
            max_value = sample_len

    avg = sum(avg) / len(avg)
    return max_value, avg


def print_dataset_statistics(
    dataset: DatasetDict, tokenizer: AutoTokenizer, split: str = "train"
) -> None:

    for k in dataset[split].features.keys():
        output = tokenizer(dataset[split][k])

        max_value, avg = compute_max_avg(output["input_ids"])

        print(f"--- {k} ---")
        print(f"Average lenght: {avg:.3f}")
        print(f"Max lenght: {max_value}")
