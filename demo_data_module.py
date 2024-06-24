# In order to run this module's function, you will need to etract the demo data
# to your local machine first.
# Please extract the news.zip in data directory.
# Make sure after extraction, this directory structure is maintained:
# /data/e/news article/News Articles/....

# To use functions from this module, import to your respective code file.
# E.g.:
# from demo_data_module import get_demo_data, count_words

# df_text = get_demo_data()
# freq_df = count_words(df_text)

# print(df_text)
# print(freq_df)


# basic imports
import os
import pandas as pd
import regex as re
from collections import Counter
import nltk
# If you encounter LookupError for stopwords, please uncomment the following line and run again
# nltk.download("stopwords")


# demo data function
def tokenize(text):
    return re.findall(r'[\w-]*\p{L}[\w-]*', text)


# demo data function
def get_demo_data():
    '''
    Returns a pd.DataFrame with 2 columns: text, tokens
    text: This column contains the original text
    tokens: This column contains a list of tokens obtained from the text
        after tokenizing it with a standard regex and removing stopwords using nltk corpus.
    '''
    # Step 1 - Get the file details
    directory = []
    file = []
    title = []
    data_text = []
    label = []
    datapath = "./data/e/news article"
    for dirname, _, filenames in os.walk(os.path.normpath(datapath), topdown=True):
        try:
            filenames.remove('README.TXT')
        except:
            pass
        for filename in filenames:
            directory.append(dirname)
            file.append(filename)
            label.append(dirname.split('\\')[-1])
            fullpathfile = os.path.join(dirname, filename)
            with open(fullpathfile, 'r', encoding="utf8", errors='ignore') as infile:
                intext = ''
                firstline = True
                for line in infile:
                    if firstline:
                        title.append(line.replace('\n', ''))
                        firstline = False
                    else:
                        intext = intext + ' ' + line.replace('\n', '')
                data_text.append(intext)

    fulldf = pd.DataFrame(list(zip(directory, file, title, data_text, label)),
                          columns=['directory', 'file', 'title', 'text', 'label'])

    df = fulldf.filter(['text'], axis=1)

    df["text"] = df["text"].apply(str.lower)
    df["tokens"] = df["text"].apply(tokenize)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords |= {"mr", "regards", "must", "would", "also", "should", "said"}
    df["tokens"] = df["tokens"].apply(
        lambda tokens: [t for t in tokens if t not in stopwords])

    return df


# demo data function
def count_words(df, column="tokens", min_freq=2):
    '''
    df: pd.DataFrame. Must include a column with given column 
        name whose values are a iterable of tokens
    column: Name of the column in the df which contains the tokens iterable
    min_freq: minimum frequency threshold. Tokens with frequency less than this will be filtered.

    Returns: a pd.DataFrame with the tokens as index and their respective frequency.
    '''
    counter = Counter()

    df[column].map(counter.update)

    freq_df = pd.DataFrame.from_dict(counter, orient="index", columns=["freq"])
    freq_df = freq_df.query('freq >= @min_freq')
    freq_df.index.name = "token"

    return freq_df.sort_values("freq", ascending=False)
