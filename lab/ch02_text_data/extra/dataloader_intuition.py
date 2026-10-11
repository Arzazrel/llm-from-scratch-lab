"""
Capitolo 2 - BONUS: intuizione sul DataLoader con dati numerici
Sliding window su sequenza di numeri per capire input/target.
"""

import importlib
import tiktoken
import torch
from importlib.metadata import version
from torch.utils.data import Dataset, DataLoader

print("torch version:", version("torch"))

# Data sampling with a sliding window with number data
# To understand the dataloader, which using a sliding window approach, more intuitive, we can consider a dataset that consists of digits only:
#
# ```
# 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 ... 1000
# ```

with open("number-data.txt", "w", encoding="utf-8") as f:
    for number in range(1001):
        f.write(f"{number} ")

# Next, we make a small modification to the `token_ids`: instead of using a tokenizer, we parse the integers directly from the text file:
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Modification
        # token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        token_ids = [int(i) for i in txt.strip().split()]

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):

    # Initialize the tokenizer
    # tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = None

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# Let's test the dataloader with a batch size of 1 for an LLM with a context size of 4:

with open("number-data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print("First batch: ",first_batch)

second_batch = next(data_iter)
print("Second batch: ",second_batch)

third_batch = next(data_iter)
print("Third batch: ",third_batch)

for batch in dataloader:
    pass

last_batch = batch
print(last_batch)

# Now, let's look at the batched inputs:

dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=False)

for inputs, targets in dataloader:
    pass

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# Finally, a data loader with shuffling:

torch.manual_seed(123)
dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=True)

for inputs, targets in dataloader:
    pass

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
