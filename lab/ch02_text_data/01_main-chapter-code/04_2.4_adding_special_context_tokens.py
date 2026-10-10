"""
Capitolo 2 - Working with text data
Sezione: 2.4 Adding special context tokens
"""
import os
import re

# ---------------------------- OLD ----------------------------
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_words = sorted(set(preprocessed))                               # sort the token set
vocab = {token:integer for integer,token in enumerate(all_words)}   # take vocabulary

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
        
# ---------------------------- NEW ----------------------------

# 2.4 Adding special context tokens

# - It's useful to add some "special" tokens for unknown words and to denote the end of a text
# - Some tokenizers use special tokens to help the LLM with additional context
# - Some of these special tokens are
#   - `[BOS]` (beginning of sequence) marks the beginning of text
#   - `[EOS]` (end of sequence) marks where the text ends (this is usually used to concatenate multiple unrelated texts, e.g., two different Wikipedia articles or two different books, and so on)
#   - `[PAD]` (padding) if we train LLMs with a batch size greater than 1 (we may include multiple texts with different lengths; with the padding token we pad the shorter texts to the longest length so that all texts have an equal length)
# - `[UNK]` to represent works that are not included in the vocabulary
#
# - Note that GPT-2 does not need any of these tokens mentioned above but only uses an `<|endoftext|>` token to reduce complexity
# - The `<|endoftext|>` is analogous to the `[EOS]` token mentioned above
# - GPT also uses the `<|endoftext|>` for padding (since we typically use a mask when training on batched inputs, we would not attend padded tokens anyways, so it does not matter what these tokens are)
# - GPT-2 does not use an `<UNK>` token for out-of-vocabulary words; instead, GPT-2 uses a byte-pair encoding (BPE) tokenizer, which breaks down words into subword units which we will discuss in a later section

# - We use the `<|endoftext|>` tokens between two independent sources of text:
# - Let's see what happens if we tokenize the following text:

tokenizer = SimpleTokenizerV1(vocab)
text = "Hello, do you like tea. Is this-- a test?"
#tokenizer.encode(text)
print("With this example text: ",text, " the encode produces an error because the word 'Hello' is not contained in the vocabulary. Add special tokens like <|unk|>.")

# - The above produces an error because the word "Hello" is not contained in the vocabulary
# - To deal with such cases, we can add special tokens like `"<|unk|>"` to the vocabulary to represent unknown words
# - Since we are already extending the vocabulary, let's add another token called `"<|endoftext|>"` which is used in GPT-2 training to denote the end of a text (and it's also used between concatenated text, like if our training datasets consists of multiple articles, books, etc.)

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
len(vocab.items())

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# - We also need to adjust the tokenizer accordingly so that it knows when and how to use the new `<unk>` token
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # assign the unknown label for the unknown token 
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]    # from voc. word to ids
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

# Let's try to tokenize text with the modified tokenizer:
tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

print("New text to tokenize: ", text)
print("Enconde, list of ids: ", tokenizer.encode(text))
print("Decode, list of words", tokenizer.decode(tokenizer.encode(text)))

