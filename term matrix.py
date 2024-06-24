import pandas as pd
import numpy as np
sentences = ["It was the best of times", 
             "it was the worst of times", 
             "it was the age of wisdom", 
             "it was the age of foolishness"]

tokenized_sentences = [[t for t in sentence.split()] for sentence in sentences]

vocabulary = set([w for s in tokenized_sentences for w in s])
[[w, i] for i,w in enumerate(vocabulary)]
def onehot_encode(tokenized_sentence):
    return [1 if w in tokenized_sentence else 0 for w in vocabulary]

onehot = [onehot_encode(tokenized_sentence) for tokenized_sentence in tokenized_sentences]

for (sentence, oh) in zip(sentences, onehot):
    print("%s: %s" % (oh, sentence))
pd.DataFrame(onehot, columns=vocabulary)
sim = [onehot[0][i] & onehot[1][i] for i in range(0, len(vocabulary))]
sum(sim)
np.dot(onehot[0], onehot[1])
np.dot(onehot, onehot[1])
onehot_encode("the age of wisdom is the best of times".split())
onehot_encode("John likes to watch movies. Mary likes movies too.".split())
onehot
