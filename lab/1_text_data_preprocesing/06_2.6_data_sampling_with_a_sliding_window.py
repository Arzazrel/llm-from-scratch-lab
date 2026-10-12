"""
Capitolo 2 - Working with text data
Sezione: 2.6 Data sampling with a sliding window
"""

import importlib
import tiktoken
import torch
from importlib.metadata import version
from torch.utils.data import Dataset, DataLoader

# ---------------------------- OLD ----------------------------
print("tiktoken version:", importlib.metadata.version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

print("Use of BPE tokenizer. The text are:\n", text)

# - BPE tokenizers break down unknown words into subwords and individual characters:
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})    
print("Enconde, list of ids: ", integers)

strings = tokenizer.decode(integers)
print("Decode, list of words: '", strings, "'")

# ---------------------------- NEW ----------------------------
# Data sampling with a sliding window
# - We train LLMs to generate one word at a time, so we want to prepare the training data accordingly where the next word in a sequence represents the target to predict:
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# - For each text chunk, we want the inputs and targets
# - Since we want the model to predict the next word, the targets are the inputs shifted by one position to the right

enc_sample = enc_text[50:]              # remove the first 50 chars

context_size = 4                        # define the window
x = enc_sample[:context_size]           # get the input window
y = enc_sample[1:context_size+1]        # get the result window

print(f"The first data window, x: {x}")
print(f"The first label window, y:      {y}")

# - One by one, the prediction would look like as follows:
print("Show an example for the input-lable pair for the training with window 4:")
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)

print("Now the same ex. but with token:")
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# - We will take care of the next-word prediction in a later chapter after we covered the attention mechanism
# - For now, we implement a simple data loader that iterates over the input dataset and returns the inputs and targets shifted by one
print("PyTorch version:", torch.__version__)

# - We use a sliding window approach, changing the position by +1:
# - Create dataset and dataloader that extract chunks from the input text dataset
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length. Stride is the step between the windows
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# function to obtain the data from dataset in window
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,num_workers=0):
    
    tokenizer = tiktoken.get_encoding("gpt2")                   # Initialize the tokenizer
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  # Create dataset

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# - Let's test the dataloader with a batch size of 1 for an LLM with a context size of 4:
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

print("\nUse new data loader")
data_iter = iter(dataloader)
first_batch = next(data_iter)
print("This is the first batch: ", first_batch)

second_batch = next(data_iter)
print("This is the second batch: ",second_batch)

# - An example using stride equal to the context length (here: 4) as shown below:
# - We can also create batched outputs
# - Note that we increase the stride here so that we don't have overlaps between the batches, since more overlap could lead to increased overfitting
print("Now set the batch size equal 8")
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
