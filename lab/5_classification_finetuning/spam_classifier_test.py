# file to test spam classification training with user's input

import tiktoken
import torch
# Import from local files
from gpt_spam_class_finetune import GPTModel
from gpt_spam_class_finetune import load_model
from gpt_spam_class_finetune import classify_review

def define_model():
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

    BASE_CONFIG.update(model_configs["gpt2-small (124M)"])  # load the small structure (the used one)
    model = GPTModel(BASE_CONFIG)                           # base general model
    # update the base model for classification
    num_classes = 2                                         # yes or false (use 2 neurons to use softmax and cross-entropy)
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)  # add last classification layer to model
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use GPU if possible
tokenizer = tiktoken.get_encoding("gpt2")
model = define_model()
model = load_model("spam_classifier.pth", model, device)                # load teh trained model
model.to(device)
        
print("Test the trained spam-classifier model.")
number_test = int(input("Insert the number of input to test: "))
if number_test < 0:
    print("Incorrect number of input.")
    exit()

for i in range(number_test):
    input_text = str(input("\nInsert the text to test: "))
    print("Input: ", input_text,"\nOutput: ", classify_review(input_text, model, tokenizer, device, max_length=len(input_text)))

# Use code adapted from Sebastian Raschka, "Build a Large Language Model (From Scratch)", Apache 2.0. Modified for personal study.