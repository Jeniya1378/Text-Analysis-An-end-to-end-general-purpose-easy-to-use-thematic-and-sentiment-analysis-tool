import regex as re
from collections import Counter
def tokenize(text):
    return re.findall(r'[\w-]*\p{L}[\w-]*', text)
text = "She likes my cats and my cats like my sofa."
tokens = tokenize(text)
print("|".join(tokens))
counter = Counter(tokens)
print(counter)
