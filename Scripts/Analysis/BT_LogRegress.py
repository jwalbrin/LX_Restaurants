# Logisitic regression 
# use either topic probabilites or sentence embeddings as features

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, multilabel_confusion_matrix

import sys
scripts_path = "/home/jon/GitRepos/LX_Restaurants/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *

topic_prob_stem = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" + 
                   "All_LX_Reviews_standard_all-MiniLM-L6-v2_%s_ProbMat.npy")
tr_split = 50

doc_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/" + 
            "Review_Data.pickle")

embed_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" +
              "Embeddings/All_LX_Review_Embeddings_all-MiniLM-L6-v2.npy")

#--- MAIN

# Get strat split training indices
if tr_split == 75:
    tr_i, te_i = strat_split_by_rating_75(doc_path)
elif tr_split == 50:
    tr_i, te_i = strat_split_by_rating_50(doc_path)

# Custom CV
custom_cv = (tr_i, te_i)
custom_cv = [(tr_i, te_i)]

    
# Load docs and embeddings
docs = load_pickled_df(doc_path)
embeddings = np.load(embed_path)
    
# Load and assign train, test features to all_features
temp_feats = np.load(topic_prob_stem % ("Train_%i" % tr_split))
feats = np.zeros((len(docs), temp_feats.shape[1]))
feats[tr_i,:] = temp_feats
temp_feats = np.load(topic_prob_stem % ("Test_%i" % (100 - tr_split)))
feats[te_i,:] = temp_feats
del temp_feats

# Safety check lengths of docs, embeddings, feats
if (len(set([len(feats), 
             len(docs), 
             len(embeddings)])) != 1):    
    raise Exception("Docs length and indices length do not match")

# Targets
y = np.array(docs.RevRating)
del docs

# pipeline

pipe = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegressionCV(cv=custom_cv, random_state=0,
                                     penalty = "l2"))])
pipe.fit(feats, y)
pipe.score(feats,y)
a = balanced_accuracy_score(y, pipe.predict(feats))
b = multilabel_confusion_matrix(y, pipe.predict(feats))

# Try embeddings

pipe.fit(embeddings, y)
pipe.score(embeddings,y)
a = balanced_accuracy_score(y, pipe.predict(embeddings))
b = multilabel_confusion_matrix(y, pipe.predict(embeddings))




# Logistic regression
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(cv=custom_cv, random_state=0,
                           penalty = "l1",
                           scoring= balanced_accuracy_score(y_true, y_pred)).fit(feats, y)

clf.predict(feats[30:60, :])
clf.predict_proba(feats[:2, :])
clf.score(feats, y)


X, y = load_iris(return_X_y=True)
clf = LogisticRegressionCV(random_state=0).fit(X, y)

clf.predict(X[:2, :])
 clf.predict_proba(X[:2, :])
 clf.score(X, y)



