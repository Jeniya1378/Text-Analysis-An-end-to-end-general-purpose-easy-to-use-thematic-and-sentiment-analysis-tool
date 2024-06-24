# basic imports
import os
import pandas as pd
import numpy as np
import ktrain
from ktrain import text

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

df = fulldf.filter(['text', 'label'], axis=1)

print("FullDf : ", fulldf.shape)
print("DF : ", df.shape)

df = pd.concat([df, df.label.astype('str').str.get_dummies()],
               axis=1, sort=False)
df = df[['text', 'sport', 'business', 'politics', 'tech', 'entertainment']]
df.head(5)

(x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(df,
                                                                   'text',  # name of column containing review text
                                                                   label_columns=[
                                                                       'sport', 'business', 'politics', 'tech', 'entertainment'],
                                                                   maxlen=500,
                                                                   preprocess_mode='bert',
                                                                   val_pct=0.1,)

# Loading a pre trained BERT and wrapping it in a ktrain.learner object                                                               
model=text.text_classifier('bert', (x_train, y_train), preproc=preproc)
learner=ktrain.get_learner(model, train_data=(
    x_train, y_train), val_data=(x_test, y_test), batch_size=2)

# Training and Tuning the model's parameters
learner.fit_onecycle(2e-1, 1)

# Inspecting Misclassifications
learner.view_top_losses(n=2, preproc=preproc)

# Save our Predictor for Later Deployment
p=ktrain.get_predictor(learner.model, preproc)
p.save('./model/text_classifier')
