# Adapted from Sebastian Raschka, "Build a Large Language Model (From Scratch)", Apache 2.0. Modified for personal study.

import argparse
import urllib.request
import zipfile
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# Import from local files
from LLM_model_done import GPTModel
from LLM_model_done import load_weights_into_gpt
from LLM_model_done import download_and_load_gpt2

# function to download and manage the dataset to fintuning the LLM to perform spam classification.
# the dataset is saved as a tab-separated text file.
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path, test_mode=False):
    # control check
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # try multiple times since CI sometimes has connectivity issues
    if test_mode:           
        max_retries = 5     # max number of retry
        delay = 5           # delay between retries in seconds
        for attempt in range(max_retries):
            try:
                # downloading the file
                with urllib.request.urlopen(url, timeout=10) as response:
                    with open(zip_path, "wb") as out_file:
                        out_file.write(response.read())
                break       # if download is successful, break out of the loop
            except urllib.error.URLError as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)           # wait before retrying
                else:
                    print("Failed to download file after several attempts.")
                    return                      # exit if all retries fail

    else:
        # downloading the file
        with urllib.request.urlopen(url) as response:
            with open(zip_path, "wb") as out_file:
                out_file.write(response.read())

    # unzipping the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # add .tsv file extension
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

# function to balance the dataset, remove the sample of the major class. Undersample the dataset so that it contains 747 instances from each class
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]                       # count the instances of "spam"
    # randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])    # Combine ham "subset" with "spam"

    return balanced_df

# function that randomly split and return train, vidation, test sets. Indicate the percentage [0,1] for training and validation set (test is the difference)
def random_split(df, train_frac, validation_frac): 
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)     # shuffle the entire DataFrame

    # calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df

# identifies the longest sequence in the training dataset and adds the padding token [endoftext token] to the others to match that sequence length
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)                                           # get data
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]] # pre-tokenize texts

        if max_length is None:
            self.max_length = self._longest_encoded_length()        # take the max length
        else:
            self.max_length = max_length
            # truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    # return sample and related label
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    # return the max length of text
    def _longest_encoded_length(self):
        max_length = 0
        # iterate for each text
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length     # set max length
        return max_length                       # return the max length

# function to calculate classification accuracy (percentage of correct predictions in a given dataset) 
# apply the `argmax`-based prediction code to all examples in a dataset and calculate the fraction of correct predictions as follows:
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

# The `calc_loss_batch` function is only interested in optimizing the last token `model(input_batch)[:, -1, :]` instead of all tokens `model(input_batch)`. 
# Now the important generated text is the last word 'yes' or 'not'
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]                                       # logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    # reduce the number of batches to match the total number of batches in the data loader
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:               # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# use the simple one
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,eval_freq, eval_iter, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                         # Calculate loss gradients
            optimizer.step()                        # Update model weights using loss gradients
            examples_seen += input_batch.shape[0]   # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1} (Batch step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

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

    return "spam" if predicted_label == 1 else "not spam"   # return the classified result


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    # plt.show()

# function to try the fine-tuning model
def try_fine_tuning_model(model, tokenizer, device, train_dataset):
    print("Test the fine-tuning model with mail on datasset:")
    
    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )

    print("First input: ", text_1,"\nFirst output: ", classify_review(text_1, model, tokenizer, device, max_length=train_dataset.max_length))

    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print("Second input: ", text_2,"\nSecond output: ", classify_review(text_2, model, tokenizer, device, max_length=train_dataset.max_length))

# function to save the fine-tuned model
def save_model():
    torch.save(model.state_dict(), "spam_classifier.pth")
    print("Save the model with name: spam_classifier.pth")

# function to load a saved model
def load_model(name, model, device):
    model_state_dict = torch.load(name, map_location=device)
    model.load_state_dict(model_state_dict)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a GPT model for classification")
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("This flag runs the model in test mode for internal testing purposes. ")
    )
    args = parser.parse_args()

    # -- Download and prepare dataset --
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path, test_mode=args.test_mode)
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    print("Save the train and test sets.")
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    # -- Create data loaders --
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset(csv_file="train.csv",max_length=None,tokenizer=tokenizer)                           # train set loader
    val_dataset = SpamDataset(csv_file="validation.csv",max_length=train_dataset.max_length,tokenizer=tokenizer)    # validation set loader  
    test_dataset = SpamDataset(csv_file="test.csv",max_length=train_dataset.max_length,tokenizer=tokenizer)         # test set loader

    num_workers = 0             # parallel download
    batch_size = 8

    torch.manual_seed(123)

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


    # -- Load pretrained model (small GPT model for testing purposes) --
    if args.test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 120,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2
        }
        model = GPTModel(BASE_CONFIG)       # base general gpt model
        model.eval()                        # not update for the weights
        device = "cpu"                      # use CPU
    else:
        CHOOSE_MODEL = "gpt2-small (124M)"
        INPUT_PROMPT = "Hi my name is"

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

        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
            f"Dataset length {train_dataset.max_length} exceeds model's context "
            f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
            f"`max_length={BASE_CONFIG['context_length']}`"
        )

        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

        model = GPTModel(BASE_CONFIG)                                           # base general model
        load_weights_into_gpt(model, params)                                    # load weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use GPU if possible

    # -- Modify and pretrained model --
    for param in model.parameters():
        param.requires_grad = False

    num_classes = 2                     # yes or false (use 2 neurons to use softmax and cross-entropy)
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)  # add last classification layer to model
    model.to(device)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    # -- Finetune modified model --
    num_epochs = int(input("Insert the number of epochs: "))
    if num_epochs < 0:
        num_epochs = 5
        
    start_time = time.time()            # start training time
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    # train the model
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
        tokenizer=tokenizer
    )

    end_time = time.time()              # end training time
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # -- Plot results --
    # loss plot
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    # accuracy plot
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")
    
    # -- try the fine-tuning model --
    try_fine_tuning_model(model, tokenizer, device, train_dataset)
    save_model()