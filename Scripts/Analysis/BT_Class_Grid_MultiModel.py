"""Uses grid search to find best classification model, 
for different input topic rep_models"""

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
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, multilabel_confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
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
rep_models = ["All_LX_Reviews_standard_all-MiniLM-L6-v2",
          "All_LX_Reviews_keybert_all-MiniLM-L6-v2"]

rep_models = ["All_LX_Reviews_keybert_all-MiniLM-L6-v2"]

tr_splits = [50, 75]

rc_vals = [0, 100, 150] # zero skips, else take k clusters

feats_to_keep = np.arange(50) # "All" or np.array of indices

mod_eval_metric = "balanced_accuracy"

cv_folds = 10

# Flags
remove_outliers = 1
softmax_feats = 0

# Pipe and parameters to search over
pipe_params = [(Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
        ]),
        {"scaler": [StandardScaler(), Normalizer()],
          # "clf__class_weight": ["balanced", None]
          "clf__class_weight": [None]},
        )]

# pipe_params = [(Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', GridSearchCV(RFE(LogisticRegression(), step=10), 
#                            params, cv=3, scoring='balanced_accuracy'))
#         ]),
#         {'estimator__class_weight': ['balanced', None]},
#         "B"
#         )]


 
#--- MAIN

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
  
# Make output dir
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)    
 
# Load y (RevRatings from docs)
docs = load_pickled_df(doc_path) 
y = np.array(docs.RevRating)
del docs  

df_out = pd.DataFrame(columns = ["RepName", "TrSplit", "ClustReduc", 
                                 "Estimator","TrScore", "TeScore", 
                                 "BestEst","CVResults"])

for m_i, m in enumerate(rep_models):    
    for ts_i, ts in enumerate(tr_splits):
        
        for rc_i, rc in enumerate(rc_vals):            
           
            # Load train / test features
            if rc != 0:
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
                
            if feats_to_keep != "All":
                X_train = X_train[:,feats_to_keep]
                X_test = X_test[:,feats_to_keep]
                
            #--- Pipe-grid-fit              
            for pp_i, (pipe, params) in enumerate(pipe_params):
                
                tic = time.time()
                
                # Tidy string names
                m_name = m.split("_")[3].capitalize()  
                e_name = str(pipe.get_params("clf")["clf"]).split("(")[0]

                # Strat k fold (outer CV for train data)
                skf = StratifiedKFold(n_splits= cv_folds, 
                                      random_state = 42, 
                                      shuffle = True).split(X_train, y_train)
                
               
                # Grid search
                grid = GridSearchCV(pipe, params, cv=skf,
                                    scoring = mod_eval_metric).fit(X_train, 
                                                                      y_train)
                                                                   
           
                # Get (best) train and test score
                train_score = grid.score(X_train, y_train)
                test_score = grid.score(X_test, y_test)    
                
                # Assign train and test scores
                new_row = [m_name, ts, rc, e_name,
                           train_score, test_score,
                           grid.best_estimator_,
                           grid.cv_results_]
                df_out.loc[len(df_out)] = new_row

                print("Model: %s tr_split_%i clust_reduc_%i %s" 
                      % (m_name,ts,rc,e_name))
                print(("Time elapsed: %1.1f, best train score: %1.4f, " +
                      "best test score: %1.4f") 
                      % (time.time() - tic, train_score, test_score))
                        
#Pickle
out_name = "GridMultiModel_Class_%s.pickle" % (str(time.time()).replace(".","_"))
pickle_path = os.path.join(output_path,out_name)
with open(pickle_path,"wb") as f:
    pickle.dump(df_out, f)

# Plot
x_labels = ["%s %s TS: %i, CR: %i" % 
            (df_out.loc[i].RepName,
             df_out.loc[i].Estimator,
            df_out.loc[i].TrSplit,
            df_out.loc[i].ClustReduc)            
            for i in np.arange(len(df_out))]

fig, ax = plt.subplots(figsize=(8,5), dpi = 300)

ax.plot(np.arange(len(df_out)), "TrScore", data = df_out, c = "blue")
ax.plot(np.arange(len(df_out)), "TeScore", data = df_out, c = "red")


ax.set_xticks(np.arange(len(df_out)),
                        labels = x_labels, rotation = 90)
ax.set_ylabel(mod_eval_metric)
ax.legend(labels = ["Train", "Test"])

