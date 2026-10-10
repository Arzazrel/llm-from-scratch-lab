"""
Capitolo 2 - Working with text data
Sezione: 2.2 Tokenizing text
"""

import os
import re
import urllib.request

# 2.2 Tokenizing text
# - In this section, we tokenize text, which means breaking text into smaller units, such as individual words and punctuation characters
# - Load raw text we want to work with
# - [The Verdict by Edith Wharton](https://en.wikisource.org/wiki/The_Verdict) is a public domain short story

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Analyze 'the-verdict.txt'")
print("Total number of character:", len(raw_text))
print("Example of the text: \n",raw_text[:99])

# - The goal is to tokenize and embed this text for an LLM
# - Let's develop a simple tokenizer based on some simple sample text that we can then later apply to the text above
# - The following regular expression will split on whitespaces
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)

print("Example test: ", text, "\nExample token: ",result)

# - We don't only want to split on whitespaces but also commas and periods, so let's modify the regular expression to do that as well
result = re.split(r'([,.]|\s)', text)
print("Redo: ",result)

# - As we can see, this creates empty strings, let's remove them
# Strip whitespace from each item and then filter out any empty strings.
result = [item for item in result if item.strip()]
print("Remove empty strings: ",result)

# - This looks pretty good, but let's also handle other types of punctuation, such as periods, question marks, and so on
text = "Hello, world. Is this-- a test?"                    # new exmple test
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)          # process
result = [item.strip() for item in result if item.strip()]  # remove empty strings
print("Example test: ", text, "\nExample token: ",result)

# - This is pretty good, and we are now ready to apply this tokenization to the raw text
print("Apply tokenization to the verdict text")
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("Example of the text: \n", preprocessed[:30])

# - Let's calculate the total number of tokens
print("Total number of the tokens: ",len(preprocessed))
