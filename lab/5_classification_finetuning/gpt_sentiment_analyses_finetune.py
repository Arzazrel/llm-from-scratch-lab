# Adapted from Sebastian Raschka, "Build a Large Language Model (From Scratch)", Apache 2.0. Modified for personal study.

import os
import sys
import tarfile
import time
import argparse
import tiktoken
from pathlib import Path
import torch
import urllib.request
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# Import from local files
from LLM_model_done import GPTModel
from LLM_model_done import load_weights_into_gpt
from LLM_model_done import download_and_load_gpt2

# ---------------------- loading dataset ----------------------
# use this dataset -> "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# 
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
    else:
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        percent = count * block_size * 100 / total_size

        speed = int(progress_size / (1024 * duration)) if duration else 0
        sys.stdout.write(
            f"\r{int(percent)}% | {progress_size / (1024**2):.2f} MB "
            f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
        )
        sys.stdout.flush()


def download_and_extract_dataset(dataset_url, target_file, directory):
    if not os.path.exists(directory):
        if os.path.exists(target_file):
            os.remove(target_file)
        urllib.request.urlretrieve(dataset_url, target_file, reporthook)
        print("\nExtracting dataset ...")
        with tarfile.open(target_file, "r:gz") as tar:
            tar.extractall()
    else:
        print(f"Directory `{directory}` already exists. Skipping download.")


def load_dataset_to_dataframe(basepath="aclImdb", labels={"pos": 1, "neg": 0}):
    data_frames = []  # List to store each chunk of DataFrame
    for subset in ("test", "train"):
        for label in ("pos", "neg"):
            path = os.path.join(basepath, subset, label)
            for file in sorted(os.listdir(path)):
                with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
                    # Create a DataFrame for each file and add it to the list
                    data_frames.append(pd.DataFrame({"text": [infile.read()], "label": [labels[label]]}))
    # Concatenate all DataFrame chunks together
    df = pd.concat(data_frames, ignore_index=True)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # Shuffle the DataFrame
    return df


def partition_and_save(df, sizes=(35000, 5000, 10000)):
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Get indices for where to split the data
    train_end = sizes[0]
    val_end = sizes[0] + sizes[1]

    # Split the DataFrame
    train = df_shuffled.iloc[:train_end]
    val = df_shuffled.iloc[train_end:val_end]
    test = df_shuffled.iloc[val_end:]

    # Save to CSV files
    train.to_csv("sent_train.csv", index=False)
    val.to_csv("sent_validation.csv", index=False)
    test.to_csv("sent_test.csv", index=False)
    
    return train, val, test

# function whic control if dataset are already loaded
def is_loaded():
    # check
    if os.path.exists("sent_train.csv") and os.path.exists("sent_validation.csv") and os.path.exists("sent_test.csv"):
        print("Dataset already loaded.")
        # load dataset
        train_df = pd.read_csv("sent_train.csv")
        val_df = pd.read_csv("sent_validation.csv")
        test_df = pd.read_csv("sent_test.csv")
    else:
        dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        print("Downloading dataset ...")
        download_and_extract_dataset(dataset_url, "aclImdb_v1.tar.gz", "aclImdb")
        print("Creating data frames ...")
        df = load_dataset_to_dataframe()
        print("Partitioning and saving data frames ...")
        train_df, val_df, test_df = partition_and_save(df)
    
    return train_df, val_df, test_df

# 
class IMDBDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]
            for text in self.data["text"]
        ]
        # Pad sequences to the longest sequence
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))
            for et in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        max_length = 0
        for text in self.data["text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

# --------------------------------------------

# ---------------------- train and validate ----------------------
def calc_loss_batch(input_batch, target_batch, model, device, trainable_token=-1):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, trainable_token, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None, trainable_token=-1):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device, trainable_token=trainable_token)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad()  # Disable gradient tracking for efficiency
def calc_accuracy_loader(data_loader, model, device, num_batches=None, trainable_token=-1):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)[:, trainable_token, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device, eval_iter, trainable_token=-1):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter, trainable_token=trainable_token)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter, trainable_token=trainable_token)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,eval_freq, eval_iter, max_steps=None, trainable_token=-1):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device, trainable_token=trainable_token)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter, trainable_token=trainable_token)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:
                break

        # New: Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter, trainable_token=trainable_token)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter, trainable_token=trainable_token)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if max_steps is not None and global_step > max_steps:
            break
            
    return train_losses, val_losses, train_accs, val_accs, examples_seen

# --------------------------------------------

# ---------------------- test and save ----------------------

# function to try the fine-tuning model
def try_fine_tuning_model(model, tokenizer, device, train_loader, val_loader, test_loader):
    print("\nEvaluating on the full datasets ...\n")

    train_accuracy = calc_accuracy_loader(train_loader, model, device, trainable_token=args.trainable_token)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, trainable_token=args.trainable_token)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, trainable_token=args.trainable_token)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
# function implements the data preprocessing steps similar to the `SpamDataset` than perform inference with spam fine-tuning model and returns the results.
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    input_ids = input_ids[:min(max_length, supported_context_length)]   # truncate sequences if they too long
    input_ids += [pad_token_id] * (max_length - len(input_ids))         # pad sequences to the longest sequence
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # add batch dimension

    # model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]              # logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()   # get the most likely class

    return "positive" if predicted_label == 1 else "negative"   # return the classified result

# function to save the fine-tuned model
def save_model():
    torch.save(model.state_dict(), "sentiment_classifier.pth")
    print("Save the model with name: sentiment_classifier.pth")

# function to load a saved model
def load_model(name, model, device):
    model_state_dict = torch.load(name, map_location=device)
    model.load_state_dict(model_state_dict)
    return model

# --------------------------------------------

# function to instantiate the chosen model
def instantiate_model(choose_model, load_weights):

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[choose_model])

    if not load_weights:
        torch.manual_seed(123)
    model = GPTModel(BASE_CONFIG)       # base model not trained

    if load_weights:                    # download the weight 
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        load_weights_into_gpt(model, params)                        # get pre-trained model

    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        default="gpt2-small (124M)",
        help=(
            "Which GPT model to use. Options: 'gpt2-small (124M)', 'gpt2-medium (355M)',"
            " 'gpt2-large (774M)', 'gpt2-xl (1558M)'."
        )
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrained",
        help=(
            "Whether to use 'pretrained' or 'random' weights."
        )
    )
    parser.add_argument(
        "--trainable_layers",
        type=str,
        default="last_block",
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_layer'."
        )
    )
    parser.add_argument(
        "--trainable_token",
        type=str,
        default="last",
        help=(
            "Which token to train. Options: 'first', 'last'."
        )
    )
    parser.add_argument(
        "--context_length",
        type=str,
        default="256",
        help=(
            "The context length of the data inputs."
            "Options: 'longest_training_example', 'model_context_length' or integer value."
        )
    )
    args = parser.parse_args()

    # -- get parameters --
    if args.trainable_token == "first":
        args.trainable_token = 0
    elif args.trainable_token == "last":
        args.trainable_token = -1
    else:
        raise ValueError("Invalid --trainable_token argument")
        
    if args.weights == "pretrained":
        load_weights = True                         # pretrain model
    elif args.weights == "random":
        load_weights = False                        # not trained model
    else:
        raise ValueError("Invalid --weights argument.")
    
    # -- define model --
    model = instantiate_model(args.model_size, load_weights)
    for param in model.parameters():
        param.requires_grad = False

    if args.model_size == "gpt2-small (124M)":
        in_features = 768
    elif args.model_size == "gpt2-medium (355M)":
        in_features = 1024
    elif args.model_size == "gpt2-large (774M)":
        in_features = 1280
    elif args.model_size == "gpt2-xl (1558M)":
        in_features = 1600
    else:
        raise ValueError("Invalid --model_size argument")

    model.out_head = torch.nn.Linear(in_features=in_features, out_features=2)   # add the last layer for classification

    if args.trainable_layers == "last_layer":
        pass
    elif args.trainable_layers == "last_block":
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
    elif args.trainable_layers == "all":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Invalid --trainable_layers argument.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -- instantiate dataloaders --

    is_loaded()
    base_path = Path(".")
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = None
    if args.context_length == "model_context_length":
        max_length = model.pos_emb.weight.shape[0]
    elif args.context_length == "longest_training_example":
        train_dataset = IMDBDataset(base_path / "train.csv", max_length=None, tokenizer=tokenizer)
        max_length = train_dataset.max_length
    else:
        try:
            max_length = int(args.context_length)
        except ValueError:
            raise ValueError("Invalid --context_length argument")

    if train_dataset is None:
        train_dataset = IMDBDataset(base_path / "sent_train.csv", max_length=max_length, tokenizer=tokenizer)
    val_dataset = IMDBDataset(base_path / "sent_validation.csv", max_length=max_length, tokenizer=tokenizer)
    test_dataset = IMDBDataset(base_path / "sent_test.csv", max_length=max_length, tokenizer=tokenizer)

    num_workers = 4                 # multi thread load
    batch_size = 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    # -- Finetune modified model --
    start_time = time.time()            # start training time
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = int(input("Insert the number of epochs: "))
    if num_epochs < 0:
        num_epochs = 5
        
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=20,
        max_steps=None, trainable_token=args.trainable_token
    )

    end_time = time.time()              # end training time
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # -- evaluate model --
    try_fine_tuning_model(model, tokenizer, device, train_loader, val_loader, test_loader)
    save_model()