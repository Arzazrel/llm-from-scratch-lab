"""
Capitolo 2 - Working with text data
Sezione: 2.3 Converting tokens into token IDs
"""
import os
import re

# 2.3 Converting tokens into token IDs
# - Next, we convert the text tokens into token IDs that we can process via embedding layers later
# - From these tokens, we can now build a vocabulary that consists of all the unique tokens

# ---------------------------- OLD ----------------------------
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# ---------------------------- NEW ----------------------------
all_words = sorted(set(preprocessed))   # sort the token set
vocab_size = len(all_words)             # take the number of the token

print("Dimension of the vocabolary if hte text: ",vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}

# - Below are the first 50 entries in this vocabulary:
print("Show the first 50 entries:")
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# - Below, we illustrate the tokenization of a short sample text using a small vocabulary:
# - Putting it now all together into a tokenizer class
# - The `encode` function turns text into token IDs
# - The `decode` function turns token IDs back into text
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# - We can use the tokenizer to encode (that is, tokenize) texts into integers
# - These integers can then be embedded (later) as input of/for the LLM
tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print("Simple example: from this text ",text," the related ids are: ",ids)

# - We can decode the integers back into text
tokenizer.decode(ids)
tokenizer.decode(tokenizer.encode(text))
