import numpy as np
from scipy.linalg import svd
import re
import numpy as np
import pandas as pd
from scipy import linalg, spatial
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer, TfidfVectorizer)
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from nltk.tokenize import word_tokenize
import nltk
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords |= {"mr", "regards", "must", "would", "also", "should", "said"}
A = np.array([[1, 2], [3, 4], [5, 6]])
print(A)
U, S, VT = svd(A)
print(U)
print(S)
print(VT)
corpus = [

          "With all of the critical success Downey had experienced throughout his career, he had not appeared in a blockbuster film. That changed in 2008 when Downey starred in two critically and commercially successful films, Iron Man and Tropic Thunder. In the article Ben Stiller wrote for Downey's entry in the 2008 edition of The Time 100, he offered an observation on Downey's commercially successful summer at the box office.",
          "On June 14, 2010, Downey and his wife Susan opened their own production company called Team Downey. Their first project was The Judge.",
          "Robert John Downey Jr. is an American actor, producer, and singer. His career has been characterized by critical and popular success in his youth, followed by a period of substance abuse and legal troubles, before a resurgence of commercial success in middle age.",
          "In 2008, Downey was named by Time magazine among the 100 most influential people in the world, and from 2013 to 2015, he was listed by Forbes as Hollywood's highest-paid actor. His films have grossed over $14.4 billion worldwide, making him the second highest-grossing box-office star of all time."
          
          ]

stop_words = set(stopwords.words('english'))
filtered_document= []
filtered_text = []

for document in corpus:
    
    clean_document = " ".join(re.sub(r"[^A-Za-z \—]+", " ", document).split())
    
    document_tokens = word_tokenize(clean_document)

    for word in document_tokens:
        if word not in stop_words:
            filtered_document.append(word)

    filtered_text.append(' '.join(filtered_document))
vectorizer = CountVectorizer()

counts_matrix = vectorizer.fit_transform(filtered_text)

feature_names = vectorizer.get_feature_names()

count_matrix_df = pd.DataFrame(counts_matrix.toarray(), columns=feature_names)
count_matrix_df.index = ['Document 1','Document 2','Document 3','Document 4']

print("Word frequency matrix: \n", count_matrix_df)
vectorizer = TfidfVectorizer(stop_words=stop_words,max_features=10000, max_df = 0.5,
                                    use_idf = True,
                                    ngram_range=(1,3))

X = vectorizer.fit_transform(filtered_text)
print(X.shape)
print(feature_names)

num_clusters = 4

km = KMeans(n_clusters=num_clusters)
km.fit(X)
    
clusters = km.labels_.tolist()
print(clusters)
U, Sigma, VT = randomized_svd(X, n_components=10, n_iter=100, random_state=122)

svd_model = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=100, random_state=122)

svd_model.fit(X)
    
print(U.shape)

for i, comp in enumerate(VT):
    terms_comp = zip(feature_names, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    print("Cluster "+str(i)+": ")
    for t in sorted_terms:
        print(t[0])
    print(" ")
