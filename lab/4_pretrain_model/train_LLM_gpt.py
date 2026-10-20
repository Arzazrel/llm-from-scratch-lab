# Adapted from Sebastian Raschka, "Build a Large Language Model (From Scratch)", Apache 2.0. Modified for personal study.

import matplotlib.pyplot as plt
import os
import torch
import urllib.request
import tiktoken

from LLM_model_done import GPTModel
from LLM_model_done import print_gpt_number_parameters
from LLM_model_done import generate_text_simple
from LLM_model_done import text_to_token_ids
from LLM_model_done import token_ids_to_text
from LLM_model_done import create_dataloader_v1
from LLM_model_done import read_show_info_text_file

# utility function to calculate the cross-entropy loss of a given batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# utility function to compute the loss for a user-specified number of batches in a data loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# function to evaluate the model
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()            # no dropout
    with torch.no_grad():   # no weights update
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# function to generate and print from input text
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded,max_new_tokens=50, context_size=context_size)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print("Input text: ", start_context)
        print("Output text: ",decoded_text.replace("\n", " "))  # print
    model.train()

# function to train the model
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,eval_freq, eval_iter, start_context, tokenizer):
    # initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0                 # number of input token processed
    global_step = -1                # number of training step (batch)

    # training loop
    for epoch in range(num_epochs):
        model.train()               # set training mode
        
        # iterate across all input batch
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                                               # reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)    # calculate loss for the batch
            loss.backward()                                                     # calculate loss gradients
            optimizer.step()                                                    # update model weights using loss gradients
            tokens_seen += input_batch.numel()      # update number of token
            global_step += 1                        # update trained batch

            # optional evaluation step, at each 'eval_freq' steps
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)   # evaluate model
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
   
        generate_and_print_sample(model, tokenizer, device, start_context)      # print a sample text after each epoch

    return train_losses, val_losses, track_tokens_seen

# function to plot the training and validation loss
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # create a second x-axis for tokens seen
    ax2 = ax1.twiny()                               # create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)    # invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()                              # adjust layout
    #plt.savefig("loss-plot.pdf")
    plt.show()

# main function
def main(gpt_config, settings):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use GPU if there is
    tokenizer = tiktoken.get_encoding("gpt2")                               # get tokenizer
    text_data = read_show_info_text_file(tokenizer)                         # get text

    model = GPTModel(gpt_config)
    model.to(device)                        # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])

    # Train/validation ratio
    train_ratio = 0.90                              # training set is 90% data
    split_idx = int(train_ratio * len(text_data))   # get the index of the split

    # load data for the training set
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    # load data for the validation set
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # Train model
    print("\nTrain the model:")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Hi my name is", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    # settings for training
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

    epochs = int(input("Insert the number of epochs for training: "))
    if epochs < 0:
        print("Invalid number of epochs. Use 10.")
        epochs = 10
        
    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": epochs,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    print("The settings for the training are: ", GPT_CONFIG_124M, OTHER_SETTINGS)
    print("Start training: ") 
    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)    # training

    print("Plots the results after training.")
    # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth"))
