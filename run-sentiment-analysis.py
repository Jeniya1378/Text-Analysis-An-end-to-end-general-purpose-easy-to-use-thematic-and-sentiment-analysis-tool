import ktrain
import requests
import os
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import flair


def sentiment_vader(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = "Positive"

    elif sentiment_dict['compound'] <= - 0.05 :
        overall_sentiment = "Negative"

    else :
        overall_sentiment = "Neutral"
  
    return negative, neutral, positive, compound, overall_sentiment

def sentiment_texblob(row):
  
    classifier = TextBlob(row)
    polarity = classifier.sentiment.polarity
    subjectivity = classifier.sentiment.subjectivity
    
    return polarity,subjectivity

def sentiment_flair(sentence):
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    s = flair.data.Sentence(sentence)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    return total_sentiment

input_text = "I am not happy."

print("=== Sentiment Analysis from pre-trained model ===")

print("=== Text classification from pretrained model VADER ===")
print(sentiment_vader(input_text))
print("=== Text classification from pretrained model textblob Naive Bayes Analyzer – NLTK classifier trained on a movie reviews corpus ===")
print(sentiment_texblob(input_text))
print("=== Text classification from pretrained model flair based on character-level LSTM neural network  ===")
print(sentiment_flair(input_text))

print("=== Text classification from trained model ===")
p = ktrain.load_predictor('./model/sentiment')
print(p.predict(input_text))