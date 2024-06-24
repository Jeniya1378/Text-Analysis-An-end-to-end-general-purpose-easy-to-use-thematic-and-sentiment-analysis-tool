import tensorflow as tf
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
import tensorflow as tf

# Loading the train dataset

data_train = pd.read_excel('./data/IMDB-Movie-Reviews-Large-Dataset-50k/train.xlsx', dtype = str)

#loading the test dataset

data_test = pd.read_excel('./data/IMDB-Movie-Reviews-Large-Dataset-50k/test.xlsx', dtype = str)

#dimension of the dataset

print("Size of train dataset: ",data_train.shape)
print("Size of test dataset: ",data_test.shape)

#printing last 5 rows of train dataset

data_train.tail()

#printing head rows of test dataset

data_test.head()

# text.texts_from_df return two tuples
# maxlen means it is considering that much words and rest are getting trucated
# preprocess_mode means tokenizing, embedding and transformation of text corpus(here it is considering BERT model)


(X_train, y_train), (X_test, y_test), preproc = text.texts_from_df(train_df=data_train,
                                                                   text_column = 'Reviews',
                                                                   label_columns = 'Sentiment',
                                                                   val_df = data_test,
                                                                   maxlen = 500,
                                                                   preprocess_mode = 'bert')

# name = "bert" means, here we are using BERT model.

model = text.text_classifier(name = 'bert',
                             train_data = (X_train, y_train),
                             preproc = preproc)

# Here we have taken batch size as 6 as from the documentation it is recommend to use this with maxlen as 500

learner = ktrain.get_learner(model=model, 
                             train_data=(X_train, y_train),
                             val_data = (X_test, y_test),
                             batch_size = 2)         

learner.lr_find(max_epochs=5)
learner.lr_plot()

#Essentially fit is a very basic training loop, whereas fit one cycle uses the one cycle policy callback

learner.fit_onecycle(lr = 2e-1, epochs = 1)

predictor = ktrain.get_predictor(learner.model, preproc)
predictor.save('./model/sentiment')

