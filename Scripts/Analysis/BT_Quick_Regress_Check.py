# Quick check assumptions of regression

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, multilabel_confusion_matrix
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV      # For optimization
from sklearn.feature_selection import RFE
from scipy.special import softmax

import sys
scripts_path = "/home/jon/GitRepos/LX_Restaurants/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *

main_path = "/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/"

feat_path = (main_path + "%s_%s_ProbMat.npy")

doc_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/" +
            "Review_Data.pickle")

output_path = ("/home/jon/GitRepos/LX_Restaurants/Output/RegressionModelling/")

# embed_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" +
#               "Embeddings/All_LX_Review_Embeddings_all-MiniLM-L6-v2.npy")

# Test parameters


rep_model = "All_LX_Reviews_keybert_all-MiniLM-L6-v2"
rep_model = "All_LX_Reviews_standard_all-MiniLM-L6-v2"


tr_split = 50

rc_val = 0 # zero skips, else take k clusters

remove_outliers = 1
softmax_feats = 0

# pipe_params = [(Pipeline([
#               ("scaler", StandardScaler()),
#               ("regress", (Ridge(random_state = 42)))
#               ]),
#               {"scaler": [StandardScaler(), Normalizer()],
#                 "regress__alpha": [0.01, 0.1, 1, 10, 100, 1000, 
#                                   10000, 100000, 1000000]
#               })]

#--- MAIN
m = rep_model
ts = tr_split
rc = rc_val

#--- Functions 
  
def non_outlier_idx(all_tr_path, X_train,
                all_te_path, X_test):
    
    """ Get topic labels for all training samples 
    (both from  df_di and loaded feature matrix)
    Get indices of non-outlying rows (where topic labels
    are the same for both"""
    with open(all_tr_path,"rb") as f:
        _ = pickle.load(f)
        all_topics_tr = pickle.load(f)
    all_topics_tr = all_topics_tr.Topic
    X_topics_tr = np.argmax(X_train, axis = 1)
    no_idx_tr = np.where((X_topics_tr - all_topics_tr) == 0)[0]

    
    """ As above but with test samples
    """
    all_topics_te = np.load(all_te_path)
    X_topics_te = np.argmax(X_test, axis = 1)
    no_idx_te = np.where((X_topics_te - all_topics_te) == 0)[0]

    return no_idx_tr, no_idx_te
  
# Load y (RevRatings from docs)
docs = load_pickled_df(doc_path) 
y = np.array(docs.RevRating)
del docs  

# Load train / test features
if rc_val != 0:
    X_train = np.load(feat_path % (m,
                      ("Train_%i_Reduc_%i_Clusters" % (ts, rc))))  
    X_test = np.load(feat_path % (m,
                   ("Test_%i_Reduc_%i_Clusters" % (100 - ts, rc))))
    all_tr_path = (main_path + 
                     "%s_Train_%i_Reduc_%i_Clusters_Info.pickle" % 
                     (m,ts, rc))
    
    all_te_path = (main_path + 
                   "%s_Test_%i_Reduc_%i_Clusters_BestProbVec.npy" % 
                     (m,100 - ts, rc))
    

else:            
    X_train = np.load(feat_path % (m, ("Train_%i" % ts)))
    X_test = np.load(feat_path % (m,  ("Test_%i" % (100 - ts))))
    all_tr_path = (main_path + "%s_Train_%i_Info.pickle" % (m,ts))
    all_te_path = (main_path + 
                   "%s_Test_%i_BestProbVec.npy" % (m,100 - ts))

# Split y by train and test indices
tr_i, te_i = strat_split_by_rating(doc_path,ts)
y_train = y[tr_i]
y_test = y[te_i]
del tr_i, te_i

if remove_outliers == 1:
    
    # Get non-outlier indices
    no_idx_tr, no_idx_te = non_outlier_idx(all_tr_path, 
                                           X_train, 
                                           all_te_path, 
                                           X_test)                
    # slice X, y
    X_train = X_train[no_idx_tr,:]
    X_test = X_test[no_idx_te,:]
    y_train = y_train[no_idx_tr]
    y_test = y_test[no_idx_te]          
    
if softmax_feats == 1:
    X_train = softmax(X_train, axis = 1)
    X_test = softmax(X_test, axis = 1)
    

# Strat k fold (outer CV for train data)
skf = StratifiedKFold(n_splits= 10, 
                      random_state = 42, 
                      shuffle = True).split(X_train, y_train)

pipe = Pipeline([
              ("scaler", StandardScaler()),
              ("regress", (Ridge(random_state = 42)))
              ])
# Fit 
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
resids = y_test - pred

# Homoscedasticity 
fig, ax = plt.subplots()
ax.scatter(y_test, resids)

# Normality of residuals
fig, ax = plt.subplots()
ax.hist(resids, bins = 50)

