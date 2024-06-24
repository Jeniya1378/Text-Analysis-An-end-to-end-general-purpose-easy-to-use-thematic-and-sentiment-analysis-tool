# basic imports
import os
import pandas as pd
import numpy as np
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
    counter = Counter()

    df[column].map(counter.update)

    freq_df = pd.DataFrame.from_dict(counter, orient="index", columns=["freq"])
    freq_df = freq_df.query('freq >= @min_freq')
    freq_df.index.name = "token"

    return freq_df.sort_values("freq", ascending=False)


df_text = get_demo_data()
freq_df = count_words(df_text)


# IDF computation
def compute_idf(df: pd.DataFrame, column: str = "tokens", preprocess=None, min_count_in_df: int = 2):

    def count(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(set(tokens))

    # count tokens
    counter = Counter()
    df[column].map(count)

    # create DataFrame and compute idf
    idf_df = pd.DataFrame.from_dict(
        counter, orient="index", columns=["count_in_df"])
    idf_df = idf_df.query("count_in_df >= @min_count_in_df")
    s = np.log(len(df)/idf_df["count_in_df"])+0.1
    s.name = "idf"
    idf_df = pd.concat([idf_df, s],
                       axis=1, ignore_index=False)

    idf_df.index.name = "token"
    return idf_df


idf_df = compute_idf(df_text)

print(idf_df.head(5))

# Calculating tf-idf
# Multiply frequency with idf to get tf-idf
idf_df["tfidf"] = freq_df["freq"] * idf_df["idf"]

print(idf_df.head(5))
