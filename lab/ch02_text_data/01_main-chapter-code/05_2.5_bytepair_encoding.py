"""
Capitolo 2 - Working with text data
Sezione: 2.5 BytePair encoding
"""
import importlib
import tiktoken
from importlib.metadata import version

# 2.5 BytePair encoding
# - GPT-2 used BytePair encoding (BPE) as its tokenizer
# - it allows the model to break down words that aren't in its predefined vocabulary into smaller subword units or even individual characters, enabling it to handle out-of-vocabulary words
# - For instance, if GPT-2's vocabulary doesn't have the word "unfamiliarword," it might tokenize it as ["unfam", "iliar", "word"] or some other subword breakdown, depending on its trained BPE merges
# - The original BPE tokenizer can be found here: [https://github.com/openai/gpt-2/blob/master/src/encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
# - In this chapter, we are using the BPE tokenizer from OpenAI's open-source [tiktoken](https://github.com/openai/tiktoken) library, which implements its core algorithms in Rust to improve computational performance
# - I created a notebook in the [./bytepair_encoder](../02_bonus_bytepair-encoder) that compares these two implementations side-by-side (tiktoken was about 5x faster on the sample text)

print("tiktoken version:", importlib.metadata.version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

# - BPE tokenizers break down unknown words into subwords and individual characters:
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})    
print("Enconde, list of ids: ", integers)

strings = tokenizer.decode(integers)
print("Decode, list of words: '", strings, "'")


