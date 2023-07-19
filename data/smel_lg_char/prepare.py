"""
Prepare the SMEL dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import glob

#"+,0123456789ABCDEFNT[]|
#"vocab size: 24
#"train has 516,290,294 tokens
#"val has 57,365,589 tokens

directory = r'C:\dev\nanoGPT\data\smel_lg_char'

# Get a list of all the text files in the directory
file_list = glob.glob(os.path.join(directory, '*.txt'))

# Create an empty string to store the merged data
data = ""

# Iterate over each file and merge its contents into the string
for file_path in file_list:
    with open(file_path, 'r', encoding='utf-8') as file:               
        part =file.read()        
        data += part
        print(f"{file_path} ({len(part)/1024/1024:,.0f}mb / {len(data)/1024/1024:,.0f}mb)")

print(f"Found {len(data)/1024/1024:,.0f}mb characters. processing...")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.95)]
val_data = data[int(n*0.95):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(directory, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
