from typing import List
import uuid
from tqdm.auto import tqdm
import random
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.optim import AdamW
from torch import nn
import torch


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: List[str],
    max_input_tokens: int = 500,
    **kargs,
) -> List[str]:
    """Generate given a list of input prompts.
    All additional arguments are used as generation
    configurations.

    Args:
        model (AutoModelForCausalLM): llm model.
        tokenizer (AutoTokenizer): llm model tokenizer.
        text (List[str]): list of prompts.
        max_input_tokens (int, optional): max number of input tokens. Defaults to 500.

    Returns:
        List[str]: list of generated text. Each is without the input prompt and EOS.
    """
    device = model.device
    # Tokenize
    input_tokenized = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        padding=True,
    ).to(device)

    # Generate
    outputs = model.generate(
        input_ids=input_tokenized["input_ids"],
        attention_mask=input_tokenized["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **kargs,  # We can add all args we want
    )

    # Strip the prompt and Decode
    outputs = outputs.tolist()
    for i in range(len(outputs)):
        outputs[i] = outputs[i][len(input_tokenized["input_ids"][i]) :]

    generated_text_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Remove EOS
    generated_text_answer = [
        sentence.replace(tokenizer.eos_token, "") for sentence in generated_text_answer
    ]

    return generated_text_answer


def fine_tune_llm(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    **kwarg,
) -> AutoModelForCausalLM:
    """Fine-tune LLM model on a given training and val sets. Dataloaders should contain
    'input_ids', 'attention_mask', and 'labels' keys. It requires a configuration dictionary with the
    following keys:
            'num_epochs': number of epochs
            'lr': learning rate
            'num_warmup_steps': warm up steps
            'weight_decay': weight decay for AdamW
            'batch_size': batch size per gput
            'gradient_accumulation_steps': accumulation steps -> batch = batch_size * gradient_accumulation_steps
            'checkpoint_path': checkpoint folder
            'logs_path': logs folder
    -

    Args:
        model (AutoModelForCausalLM): llm model.
        tokenizer (AutoTokenizer): llm tokenizer.
        train_dataloader (DataLoader): train dataloader.
        val_dataloader (DataLoader): val dataloader.

    Returns:
        AutoModelForCausalLM: fine-tuned model.
    """
    config = kwarg
    device = model.device

    # Create checkpoint and log folders
    fine_tune_id = str(uuid.uuid1())
    checkpoint_path = Path(config["checkpoint_path"]).joinpath(fine_tune_id)
    logs_path = Path(config["logs_path"]).joinpath(fine_tune_id)
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    logs_path.mkdir(exist_ok=True, parents=True)

    # Instantiate optimizer and loss
    optimizer = AdamW(
        params=model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    criteria = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id
    )  # Ignore pad token

    # Instantiate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config["num_warmup_steps"],
        num_training_steps=(len(train_dataloader) * config["num_epochs"])
        // config["gradient_accumulation_steps"],
    )

    total_train_iterations = len(train_dataloader)
    total_eval_iterations = len(val_dataloader)

    # Create log writer
    writer = SummaryWriter(logs_path.as_posix())

    # Training loop
    progress_bar_train = tqdm(total=config["num_epochs"], desc="Fine-Tuning LLM")
    for epoch in range(config["num_epochs"]):
        model.train()
        progress_bar_epoch = tqdm(total=total_train_iterations)
        for step, batch in enumerate(train_dataloader):
            # Progress bar description
            if step == 0:
                desc = f"Epoch {epoch} | Loss train: N/A"
            else:
                desc = f"Epoch {epoch} | Loss train: {loss.item() * config['gradient_accumulation_steps']:.4e}"
            progress_bar_epoch.set_description(desc=desc)

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

            logits = outputs.logits[:, :-1, :]  # Remove last logit prediction
            logits = logits.reshape(-1, logits.shape[-1])  # Reshape for CE
            labels = batch["labels"][:, 1:]  # Shifts labels
            labels = labels.reshape(-1)  # Reshape for CE

            loss = criteria(logits, labels.to(device))

            # Gradient accumulation
            loss = (
                loss / config["gradient_accumulation_steps"]
            )  # NB: it changes logs scale
            loss.backward()

            # Gradient clippling
            clip_grad_norm_(model.parameters(), config["max_grad_norm"])

            if step % config["gradient_accumulation_steps"] == 0:

                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad()

                # Save TB logs
                writer.add_scalar(
                    "Train Loss",
                    loss.item() * config["gradient_accumulation_steps"],
                    global_step=(step + total_train_iterations * epoch),
                )

            # Update epoch bar
            progress_bar_epoch.update(1)

        # Evaluation loop
        model.eval()
        total_val_loss = 0.0
        progress_bar_val = tqdm(total=total_eval_iterations)
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                # Progress bar description
                if step == 0:
                    desc = f"Epoch {epoch} | Loss val avg: N/A"
                else:
                    desc = f"Epoch {epoch} | Loss val avg: {total_val_loss/step:.4e}"
                progress_bar_val.set_description(desc=desc)

                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )

                logits = outputs.logits[:, :-1, :]  # Remove last logit prediction
                logits = logits.reshape(-1, logits.shape[-1])  # Reshape for CE
                labels = batch["labels"][:, 1:]  # Reshape for CE
                labels = labels.reshape(-1)  # Reshape for CE

                loss = criteria(logits, labels.to(device))

                total_val_loss += loss.item()

                # Update epoch bar
                progress_bar_val.update(1)

        average_val_loss = total_val_loss / len(val_dataloader)

        # Save TB logs
        writer.add_scalar(
            "Val Loss",
            average_val_loss,
            global_step=(total_train_iterations + total_train_iterations * epoch),
        )

        # Saving model
        model.save_pretrained(checkpoint_path.as_posix())

        # Update train bar
        progress_bar_train.update(1)

    return model


def seed_all(seed: int) -> None:
    """Seed random, numpy, and torch
    with the given input seed.

    Args:
        seed (int): input seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
