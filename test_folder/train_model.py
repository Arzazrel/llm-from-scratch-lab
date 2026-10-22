# Adapted from Sebastian Raschka, "Build a Large Language Model (From Scratch)", Apache 2.0. Modified for personal study.
# ---------------------------------------------------------------------------
# train_model.py
# Creates, fine-tunes and saves every kind of model used in this repository,
# so that test_model.py can later load and run them without touching the lab.
#
# Supported tasks (chosen with --task):
#   gen        : simple text generation. No training: it just downloads the
#                GPT-2 weights, loads them into the base model and saves it.
#   adv_gen    : same stored weights as 'gen', but saved under the advanced
#                naming convention that embeds the default temperature and k.
#                (temperature/top-k are decoding-time options, not weights, so
#                 the network is identical; the separate name lets test_model.py
#                 pick a preset.)
#   spam_cls   : spam classifier fine-tuning (SMS Spam Collection dataset).
#   sent_cls   : sentiment classifier fine-tuning (IMDB dataset).
#   instr      : instruction-following fine-tuning (Alpaca-style dataset).
#
# Each run asks for the GPT-2 size (small/medium/large/XL) unless --size is
# given, and exposes the relevant variants (how many blocks to unfreeze for
# classification; whether to mask the instruction for 'instr') with defaults
# aligned to the lab files.
#
# Run `python train_model.py --help` for the full list of options, and see
# commands.txt for ready-made command lines.
# ---------------------------------------------------------------------------

import os
import argparse
import time
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import tiktoken
import torch
from functools import partial
from torch.utils.data import DataLoader

import model_utils as mu


# ---------------------------------------------------------------------------
# Helpers: interactive prompts with defaults
# ---------------------------------------------------------------------------

def ask_choice(prompt, choices, default):
    """Ask the user to pick one of 'choices'; empty input returns default."""
    choices_str = "/".join(choices)
    while True:
        raw = input(f"{prompt} [{choices_str}] (default: {default}): ").strip()
        if raw == "":
            return default
        if raw in choices:
            return raw
        print(f"  Invalid choice. Allowed: {choices_str}")


def ask_int(prompt, default):
    raw = input(f"{prompt} (default: {default}): ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  Not an integer, using default {default}.")
        return default


def ask_float(prompt, default):
    raw = input(f"{prompt} (default: {default}): ").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  Not a number, using default {default}.")
        return default


def ask_yes_no(prompt, default=False):
    d = "y" if default else "n"
    raw = input(f"{prompt} [y/n] (default: {d}): ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")


# ---------------------------------------------------------------------------
# Common: build base model with pretrained GPT-2 weights
# ---------------------------------------------------------------------------

def build_pretrained_base(size, models_dir="gpt2"):
    print(f"\nBuilding base GPT-2 '{size}' and loading pretrained weights ...")
    model = mu.build_base_model(size)
    mu.load_pretrained_into_model(model, size, models_dir=models_dir)
    return model


# ---------------------------------------------------------------------------
# TASK: gen / adv_gen  (no training, just load + save)
# ---------------------------------------------------------------------------

def run_generation(size, advanced, temperature, top_k):
    model = build_pretrained_base(size)
    if advanced:
        mu.save_model(model, "adv_gen", size, temperature=temperature, top_k=top_k)
    else:
        mu.save_model(model, "gen", size)
    print("Done. The generation model is ready to be used by test_model.py.")


# ---------------------------------------------------------------------------
# Classification helpers (freezing policy)
# ---------------------------------------------------------------------------

def apply_freezing(model, trainable_blocks):
    """Freeze everything, then unfreeze the classification head, the final
    norm, and the last 'trainable_blocks' transformer blocks.
    trainable_blocks = 0 -> only the new head (+ final norm) is trained
    trainable_blocks = 1 -> also the last transformer block (lab default)
    trainable_blocks = N -> the last N transformer blocks."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.out_head.parameters():
        p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True
    n = len(model.trf_blocks)
    trainable_blocks = max(0, min(trainable_blocks, n))
    for b in range(n - trainable_blocks, n):
        for p in model.trf_blocks[b].parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# TASK: spam_cls
# ---------------------------------------------------------------------------

def prepare_spam_data():
    """Download and prepare the SMS Spam Collection dataset into CSV files.
    Same source and split proportions as gpt_spam_class_finetune.py."""
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    if not data_file_path.exists():
        print("Downloading SMS Spam Collection dataset ...")
        with urllib.request.urlopen(url) as response:
            with open(zip_path, "wb") as out_file:
                out_file.write(response.read())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_path)
        original = Path(extracted_path) / "SMSSpamCollection"
        os.rename(original, data_file_path)
        print(f"Dataset saved as {data_file_path}")

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

    # balance the dataset (same approach as the lab)
    n_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(n_spam, random_state=123)
    balanced = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    balanced["Label"] = balanced["Label"].map({"ham": 0, "spam": 1})

    # split 70/10/20
    balanced = balanced.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(balanced) * 0.7)
    val_end = train_end + int(len(balanced) * 0.1)
    balanced[:train_end].to_csv("spam_train.csv", index=False)
    balanced[train_end:val_end].to_csv("spam_validation.csv", index=False)
    balanced[val_end:].to_csv("spam_test.csv", index=False)
    return "spam_train.csv", "spam_validation.csv"


def prepare_sentiment_data():
    """Prepare the IMDB sentiment dataset into CSV files.
    Mirrors gpt_sentiment_analyses_finetune.py (same source URL)."""
    train_csv, val_csv = "sent_train.csv", "sent_validation.csv"
    if os.path.exists(train_csv) and os.path.exists(val_csv):
        return train_csv, val_csv

    import tarfile
    dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = "aclImdb_v1.tar.gz"
    if not os.path.exists("aclImdb"):
        print("Downloading IMDB dataset (large, may take a while) ...")
        urllib.request.urlretrieve(dataset_url, tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall()

    rows = []
    for label_name, label in (("pos", 1), ("neg", 0)):
        for split in ("train", "test"):
            folder = Path("aclImdb") / split / label_name
            if not folder.exists():
                continue
            for file in folder.iterdir():
                rows.append({"text": file.read_text(encoding="utf-8"), "label": label})
    df = pd.DataFrame(rows).sample(frac=1, random_state=123).reset_index(drop=True)

    train_end = int(len(df) * 0.7)
    val_end = train_end + int(len(df) * 0.1)
    df[:train_end].to_csv(train_csv, index=False)
    df[train_end:val_end].to_csv(val_csv, index=False)
    df[val_end:].to_csv("sent_test.csv", index=False)
    return train_csv, val_csv


def run_classification(task, size, num_epochs, trainable_blocks, batch_size):
    tokenizer = mu.get_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if task == "spam_cls":
        train_csv, val_csv = prepare_spam_data()
        text_col, label_col = "Text", "Label"
    else:  # sent_cls
        train_csv, val_csv = prepare_sentiment_data()
        text_col, label_col = "text", "label"

    train_ds = mu.ClassificationDataset(train_csv, tokenizer, text_col, label_col, max_length=None)
    val_ds = mu.ClassificationDataset(val_csv, tokenizer, text_col, label_col, max_length=train_ds.max_length)

    torch.manual_seed(123)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    # base model + pretrained weights, then classification head + freezing
    model = build_pretrained_base(size)
    model.out_head = torch.nn.Linear(in_features=mu.get_config(size)["emb_dim"], out_features=2)
    apply_freezing(model, trainable_blocks)
    model.to(device)

    print(f"\nFine-tuning {task} (size={size}, epochs={num_epochs}, "
          f"trainable_blocks={trainable_blocks}) ...")
    start = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    mu.train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                               num_epochs=num_epochs, eval_freq=50, eval_iter=5)
    print(f"Training completed in {(time.time()-start)/60:.2f} minutes.")

    mu.save_model(model, task, size)


# ---------------------------------------------------------------------------
# TASK: instr
# ---------------------------------------------------------------------------

def prepare_instruction_data():
    file_path = "instruction-data.json"
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
           "ch07/01_main-chapter-code/instruction-data.json")
    if not os.path.exists(file_path):
        print("Downloading instruction dataset ...")
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_data)
    with open(file_path, "r") as f:
        return json.load(f)


def run_instruction(size, num_epochs, mask_instruction, batch_size):
    tokenizer = mu.get_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data = prepare_instruction_data()
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    train_data = data[:train_portion]
    val_data = data[train_portion + test_portion:]
    print(f"Training set: {len(train_data)} | Validation set: {len(val_data)}")

    # collate function; optionally also mask the instruction part in the loss
    base_collate = partial(mu.custom_collate_fn, device=device, allowed_max_length=1024)

    if mask_instruction:
        def collate(batch):
            inputs, targets = base_collate(batch)
            targets = mu.mask_instruction_in_targets(inputs, targets, tokenizer)
            return inputs, targets
    else:
        collate = base_collate

    torch.manual_seed(123)
    train_loader = DataLoader(mu.InstructionDataset(train_data, tokenizer),
                              batch_size=batch_size, collate_fn=collate,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(mu.InstructionDataset(val_data, tokenizer),
                            batch_size=batch_size, collate_fn=collate,
                            shuffle=False, drop_last=False, num_workers=0)

    model = build_pretrained_base(size)
    model.to(device)

    print(f"\nInstruction fine-tuning (size={size}, epochs={num_epochs}, "
          f"mask_instruction={mask_instruction}) ...")
    start = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    mu.train_model_simple(model, train_loader, val_loader, optimizer, device,
                          num_epochs=num_epochs, eval_freq=5, eval_iter=5, tokenizer=tokenizer)
    print(f"Training completed in {(time.time()-start)/60:.2f} minutes.")

    mu.save_model(model, "instr", size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train / fine-tune and save GPT-2 models for the different tasks.")
    parser.add_argument("--task", choices=["gen", "adv_gen", "spam_cls", "sent_cls", "instr"],
                        help="Which model to build/train. If omitted, it is asked interactively.")
    parser.add_argument("--size", choices=["small", "medium", "large", "XL"],
                        help="GPT-2 size. If omitted, it is asked interactively.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (classification/instruction).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Default temperature stored in the adv_gen model name.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Default top-k stored in the adv_gen model name.")
    parser.add_argument("--trainable_blocks", type=int, default=1,
                        help="Classification: how many last transformer blocks to unfreeze (default 1).")
    parser.add_argument("--mask_instruction", action="store_true",
                        help="Instruction: mask the instruction part so the loss is on the response only.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default 8).")
    parser.add_argument("--non_interactive", action="store_true",
                        help="Do not ask anything; rely on CLI arguments and defaults.")
    args = parser.parse_args()

    # Resolve task and size (interactively if needed)
    task = args.task
    if task is None and not args.non_interactive:
        task = ask_choice("Which model do you want to train?",
                          ["gen", "adv_gen", "spam_cls", "sent_cls", "instr"], "gen")
    task = task or "gen"

    size = args.size
    if size is None and not args.non_interactive:
        size = ask_choice("Which GPT-2 size?", ["small", "medium", "large", "XL"], "small")
    size = size or "small"

    # Dispatch
    if task in ("gen", "adv_gen"):
        advanced = (task == "adv_gen")
        temperature, top_k = args.temperature, args.top_k
        if advanced and not args.non_interactive:
            temperature = ask_float("Default temperature to store in the name", temperature)
            top_k = ask_int("Default top-k to store in the name", top_k)
        run_generation(size, advanced, temperature, top_k)

    elif task in ("spam_cls", "sent_cls"):
        epochs = args.epochs
        trainable_blocks = args.trainable_blocks
        if not args.non_interactive:
            if epochs is None:
                epochs = ask_int("Number of epochs", 5)
            trainable_blocks = ask_int(
                "How many last transformer blocks to unfreeze (0 = only head)", trainable_blocks)
        epochs = epochs if epochs is not None else 5
        run_classification(task, size, epochs, trainable_blocks, args.batch_size)

    elif task == "instr":
        epochs = args.epochs
        mask_instruction = args.mask_instruction
        if not args.non_interactive:
            if epochs is None:
                epochs = ask_int("Number of epochs", 2)
            mask_instruction = ask_yes_no(
                "Mask the instruction part (train on the response only)?", mask_instruction)
        epochs = epochs if epochs is not None else 2
        run_instruction(size, epochs, mask_instruction, args.batch_size)


if __name__ == "__main__":
    main()
