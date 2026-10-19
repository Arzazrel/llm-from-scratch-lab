# Evaluating generative text models

import torch
import torch.nn as nn
import tiktoken
from LLM_model_done import GPTModel
from LLM_model_done import print_gpt_number_parameters
from LLM_model_done import generate_text_simple
from LLM_model_done import text_to_token_ids
from LLM_model_done import token_ids_to_text

# ---- Using GPT to generate text ----
print(" ---- Simple example before the metrics and training. ----")
# Configuration details for the LLM based on the 124 million parameter GPT-2 model:
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Shortened context length (orig: 1024) to reduce the computational resource requirements for training the model
    "emb_dim": 768,         # Embedding dimension for token inputs
    "n_heads": 12,          # Number of attention heads (multi-head attention)
    "n_layers": 12,         # Number of transformer block in the model
    "drop_rate": 0.1,       # Dropout rate [0,1]. 0.1 is a common value
    "qkv_bias": False       # Query-Key-Value bias in the multi-head attention mechanism when computing query (Q), key (K), and value (V) tensors
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)       # instantiate the model
model.eval();                           # disable dropout during inference
print_gpt_number_parameters(model)      # print number of parameters

max_new_tokens = 10                 # define the max number of words to predict
start_context = "Hello, my name"    # start text example

tokenizer = tiktoken.get_encoding("gpt2")           # get tokenizer 
    
print(f"\nStart with the generation text example. Maximum number of Word to predict: {max_new_tokens:,}\nNote: the LLM is not trained.")
print("\nInput text:", start_context)

# get the predictions
generated_token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=max_new_tokens,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(generated_token_ids, tokenizer))

# ---- Calculating the text generation loss: cross-entropy and perplexity ----
print("\n---- See the metric. ----")

inputs = torch.tensor([[16833, 3626, 6100],     # ["every effort moves",
                       [40,    1107, 588]])     #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],    # [" effort moves you",
                        [1107,  588, 11311]])   #  " really like chocolate"]

print("Input example:\n", inputs,"\nTarget example:\n",targets)

# get the output from the input
with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)                  # probability of each token in vocabulary
generated_token_ids = torch.argmax(probas, dim=-1, keepdim=True)  # get the most probability token

print("Logit results: ", logits,"\nSoftmax result: ",probas,"\nArgmax result -> most probability token IDs:\n", generated_token_ids)
# decode the output
print(f"Outputs batch 1: {token_ids_to_text(generated_token_ids[0].flatten(), tokenizer)}")
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")

# token probabilities corresponding to the target indices
print("Token probabilities corresponding to the target indices:")
# input text 1
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)
# input text 2
text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

print("\nCross-entropy loss in deep learning: minimize the negative average log-probability value")
# compute logarithm of all token probabilities
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print("Logarithm of all token probabilities: ", log_probas)

avg_log_probas = torch.mean(log_probas)
neg_avg_log_probas = avg_log_probas * -1
print("Average logarithm: ",avg_log_probas,"\nNegative average logarithm: ", neg_avg_log_probas)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print("Torch cross_entropy loss: ", loss)

print("\nPerplexity: is the exponential of the cross-entropy loss.")
perplexity = torch.exp(loss)        # compute the perpexity
print("Perplexity: ",perplexity)    # SEE NOTE 0

"""
NOTE
- 0 -> perplexity provides a measure of how well the probability distribution predicted by the model matches the actual distribution of the words in the dataset. 
       Lower perplexity indicates that the model predictions are closer to the actual distribution.
"""