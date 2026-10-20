# Adapted from Sebastian Raschka, "Build a Large Language Model (From Scratch)", Apache 2.0. Modified for personal study.

import matplotlib.pyplot as plt
import tiktoken
import torch
from train_LLM_gpt import main
from LLM_model_done import generate_text_simple
from LLM_model_done import text_to_token_ids
from LLM_model_done import token_ids_to_text

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
    
def train():
    
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
    
    return model
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use GPU if there is
tokenizer = tiktoken.get_encoding("gpt2")                               # get tokenizer
input_ids = text_to_token_ids("Hi, my name is ", tokenizer).to(device)
model = train()

print("Even if we execute the `generate_text_simple` function above multiple times, the LLM will always generate the same outputs. Try multiple time:")
# the next generated token is the token corresponding to the largest probability score among all tokens in the vocabulary
model.eval()


token_ids = generate_text_simple(
    model=model,
    idx=input_ids,
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

token_ids = generate_text_simple(
    model=model,
    idx=input_ids,
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

print("\nIntroduce two concepts: *temperature scaling* and *top-k* sampling")

# Temperature scaling
print("\nTemperature scaling:")
print("Sample the next token using The `torch.multinomial(probs, num_samples=1)`, each index's chance of being picked corresponds to its probability.")
print("Create an example:")

vocab = { 
    "closer": 0,
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8,
} 
inverse_vocab = {v: k for k, v in vocab.items()}
# example of output
next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
probas = torch.softmax(next_token_logits, dim=0)        # compute the probability
next_token_id = torch.argmax(probas).item()             # get the next token
print("Example of input: ", vocab,"\nOutput: ",next_token_logits,"\nProbability: ",probas,"\nNext token: ", inverse_vocab[next_token_id])

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print("Use multinomial: ", inverse_vocab[next_token_id])

def print_sampled_tokens(probas):
    torch.manual_seed(123)                                                              # manual seed for reproducibility
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)

# We can control the distribution and selection process via a concept called temperature scaling
# - "Temperature scaling" is just a fancy word for dividing the logits by a number greater than 0
# - Temperatures greater than 1 will result in more uniformly distributed token probabilities after applying the softmax
# - Temperatures smaller than 1 will result in more confident (sharper or more peaky) distributions after applying the softmax
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

# Temperature values
temperatures = [1, 0.1, 5]  # Original, higher confidence, and lower confidence
# Calculate scaled probabilities
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
# Plotting
x = torch.arange(len(vocab))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')

ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()

plt.tight_layout()
plt.savefig("temperature-plot.pdf")
plt.show()

print("\nTop-k sampling (with k = 3):")
print("Take only from the most k probably next word with temperature sampling.")
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)

print("Top logits:", top_logits)
print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float('-inf')), 
    other=next_token_logits
)

print("New logits: ", new_logits)

topk_probas = torch.softmax(new_logits, dim=0)
print("Top-k probability: ", topk_probas)

# modifying the text generation function introducing temperature sampling and top-k sampling
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # for-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

input_ids = text_to_token_ids("Hi, my name is ", tokenizer).to(device)
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=input_ids,
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("\nUse new text generation.\nOutput text:\n", token_ids_to_text(token_ids, tokenizer))
