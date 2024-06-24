from demo_data_module import get_demo_data, count_words
from word_frequency import wordcloud  # reusing wordcloud from previous task

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords |= {"mr", "regards", "must", "would", "also", "should", "said"}


# display topics
def display_topics(model, features, max_words=5):
    '''
    A function to display features(words) & their percentage contribution to a topic.
    ---
    ### Params:
    model: A model that can do topic modelling. Like SVD, NMF, LDA etc.
    features: A numpy ndarray or a list of feature names obtained from TF-IDF Vectorizer or Count Vectorizer
    max_words: maximum number of features to display under a single topic.
    '''
    for topic, word_vector in enumerate(model.components_):
        total = word_vector.sum()
        largest = word_vector.argsort()[::-1]
        print("\nTopic %02d" % topic)
        for i in range(0, max_words):
            print(" %s (%2.2f)" %
                  (features[largest[i]], word_vector[largest[i]]*100.0/total))


# SVD word frequency
def get_svd_words_frequency(model, features, max_words):
    '''
    A function to get topic wise distribution of features(words) & their percentage contribution to a topic.
    ---
    ### Params:
    model: A model that can do topic modelling. Like SVD, NMF, LDA etc.
    features: A numpy ndarray or a list of feature names obtained from TF-IDF Vectorizer or Count Vectorizer
    max_words: maximum number of features to display under a single topic.
    ---
    ### Return: 
    A dictionary containing topics as keys and features(word) and their contribution to the topics in a dictionary format as values
    e.g.: {"topic 0": {"word 0": 0.97, "word 2": 0.98, ...}, ...}
    '''
    topics = {}
    for topicIdx, words in enumerate(model.components_):
        freq = {}
        largest = words.argsort()[::-1]
        for i in range(0, max_words):
            freq[features[largest[i]]] = abs(words[largest[i]])

        topics[topicIdx] = freq
    return topics


# Cluster Size plot
def plot_cluster_sizes(model, output_loc):
    '''
    A function which shows the size of the clusters using bar plot
    ---
    ### Params:
    model:
    output_loc: The output location of the plot. Should include <filename>.png
    '''
    sizes = []
    for i in range(model.n_clusters):
        sizes.append({"Cluster": i, "Size": np.sum(model.labels_ == i)})
    pd.DataFrame(sizes).set_index("Cluster").plot.bar(figsize=(16, 9))
    plt.savefig(output_loc)


# Cluster word frequency
def get_cluster_word_frequency(model, vectors, features, max_words):
    '''
    A function to get cluster wise distribution of features(words) & their percentage contribution to a topic.
    ---
    ### Params:
    model: A clustering model. Like K-Means.
    vectors: TF-IDF vectors of the features(words). We need this as we need to manually calculate the contribution factor
    features: A numpy ndarray or a list of feature names obtained from TF-IDF Vectorizer or Count Vectorizer
    max_words: maximum number of features to display under a single topic.
    ---
    ### Return: 
    A dictionary containing cluster as keys and features(word) and their contribution to the clusters in a dictionary format as values
    e.g.: {"cluster 0": {"word 0": 0.97, "word 2": 0.98, ...}, ...}
    '''
    clusters = {}
    for clusterIdx, cluster in enumerate(np.unique(model.labels_)):
        freq = {}
        words = vectors[model.labels_ == cluster].sum(axis=0).A[0]
        largest = words.argsort()[::-1]  # invert sort order
        for i in range(0, max_words):
            freq[features[largest[i]]] = abs(words[largest[i]])

        clusters[clusterIdx] = freq
    return clusters


def hbar_plot_word_probabilities(topics, output_loc, figsize=(12, 12)):
    '''
    A function which makes horizontal bar plots for words contribution to a particular topic
    ---
    ### Params:
    topics: A dictionary containing topics as keys and features(word) and their contribution to the topics in a dictionary format as values
    e.g.: {"topic 0": {"word 0": 0.97, "word 2": 0.98, ...}, ...}
    output_loc: Relative directory location to save the plots. No need to mention filename.png
    figsize (optional) : figure size for matplotlib pyplot
    '''
    for topic in topics:
        plt.figure(figsize=figsize)
        plt.title(f"Topic {topic}")
        plt.barh(list(topics[topic].keys()), list(topics[topic].values()))
        plt.xlabel("Contribution to the topic (%)")
        plt.grid()
        print(
            f"Saving Bar for topic {topic} to {output_loc}/Topic_{topic}.png")
        plt.savefig(os.path.join(output_loc, f"Topic_{topic}_Hbar.png"))
        plt.close()


# Usage
# Comment the following code when using this file as a module to import the functions
# load data
df_text = get_demo_data()
freq_df = count_words(df_text)

# SVD
tfidf_text = TfidfVectorizer(stop_words=stopwords, min_df=5, max_df=0.7)
vectors_text = tfidf_text.fit_transform(df_text['text'])

svd_para_model = TruncatedSVD(n_components=10, random_state=42)
W_svd_para_matrix = svd_para_model.fit_transform(vectors_text)

# Clustering
k_means = KMeans(n_clusters=10, random_state=42)
k_means.fit(vectors_text)


# Usage SVD
svd_topics_with_word_freq = get_svd_words_frequency(svd_para_model,
                                                    tfidf_text.get_feature_names_out(), 40)


display_topics(svd_para_model, tfidf_text.get_feature_names_out())

for topic in svd_topics_with_word_freq:
    wordcloud(svd_topics_with_word_freq[topic],
              f"svd_topic_{topic}.png", f"SVD Topic {topic}", max_words=100,
              wordcloud_style={"background_color": "white", "width": 960, "height": 540})  # reusing the wordcloud function from word_frequency.py


hbar_plot_word_probabilities(svd_topics_with_word_freq, "./")

# Usage Cluster

plot_cluster_sizes(k_means, "cluster_size.png")

cluster_with_word_freq = get_cluster_word_frequency(k_means, vectors_text,
                                                    tfidf_text.get_feature_names_out(), 40)


for cluster in cluster_with_word_freq:
    wordcloud(cluster_with_word_freq[cluster],
              f"cluster_{cluster}.png", f"Cluster {cluster}", max_words=100,
              wordcloud_style={"background_color": "white", "width": 960, "height": 540})  # reusing the wordcloud function from word_frequency.py
