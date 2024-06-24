import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
#%load_ext tensorboard
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
sentence = "The wide road shimmered in the hot sun While a bag-of-words model predicts a word given the neighboring context, a skip-gram model predicts the context (or neighbors) of a word, given the word itself. The model is trained on skip-grams, which are n-grams that allow tokens to be skipped (see the diagram below for an example). The context of a word can be represented through a set of skip-gram pairs of (target_word, context_word) where context_word appears in the neighboring context "
tokens = list(sentence.lower().split())
print(len(tokens))
vocab, index = {}, 1  # start indexing from 1
vocab['<pad>'] = 0  # add a padding token
for token in tokens:
  if token not in vocab:
    vocab[token] = index
    index += 1
vocab_size = len(vocab)
print(vocab)
inverse_vocab = {index: token for token, index in vocab.items()}
print(inverse_vocab)
sequence = [vocab[word] for word in tokens]
print(sequence)
