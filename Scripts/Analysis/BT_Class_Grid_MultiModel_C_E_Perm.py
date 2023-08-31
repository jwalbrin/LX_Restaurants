"""Uses grid search to find best classification model, 
for different input topic rep_models
Version C is binary HiLow classification
E = embeddings"""

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from scipy.special import softmax

import sys
scripts_path = "/home/jon/GitRepos/LX_Restaurants/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *

main_path = "/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/"

# feat_path = (main_path + "%s_%s_ProbMat.npy")
feat_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/Embeddings/" +
             "%s.npy")

doc_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/" +
            "Review_Data.pickle")

df_tpc_path = (main_path + 
               "ByClass/All_LX_Reviews_ByClass_%s_Train_%i.pickle")

df_di_path = (main_path + 
               "%s_Train_%i_Info.pickle")

output_path = ("/home/jon/GitRepos/LX_Restaurants/Output/RegressionModelling/")

# embed_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" +
#               "Embeddings/All_LX_Review_Embeddings_all-MiniLM-L6-v2.npy")

# Test parameters
# rep_model = "All_LX_Review_Embeddings_all-MiniLM-L6-v2_UMAP_50"
rep_model = "All_LX_Review_Embeddings_all-MiniLM-L6-v2"

topic_model_name = "All_LX_Reviews_standard_all-MiniLM-L6-v2"

tr_split = 75

feats_to_keep = np.arange(50) # "All" or np.array of indices
feats_to_keep = "All"

mod_eval_metric = "balanced_accuracy"

cv_folds = 10

n_perm = 3

perc_best_topic = 10 # percentage of best topics to remove

# Pipe and parameters to search over
pipe_params = [(Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter = 300, random_state = 42))
        ]),
        {"scaler": [StandardScaler(), Normalizer()],
          # "clf__class_weight": ["balanced", None]
          "clf__class_weight": [None]},
        )]

# pipe_params = [(Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', RandomForestClassifier(max_samples = 0.5, n_jobs = 8, 
#                                         random_state = 42))
#         ]),
#         {"scaler": [StandardScaler(), Normalizer()],
#           # "clf__class_weight": ["balanced", None]
#           "clf__n_estimators": [50, 200],
#           "clf__min_samples_split": [5, 10, 20]},
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

def get_perc_best_topic_idx(df_tpc,df_di, perc_best_topic, classes, hl_flag):
    """ Get the indices (of samples in df_di) for desired percentage (e.g. 10%) 
    of samples that are most likely to belong to topics by a given rule
    (e.g. topics determined by High > low frequency)
    df_tpc = df of topics per class
    df_di = df of doc info for all samples in given split
    perc_best_topic = percentage of samples to obtain
    classes = list of classes of len(df_di)
    hl_flag = flag for asc or desc sorting of High_OVer_low 
        i.e. 1 = high > low, 0 low > high
    """    
   
    """Get CumSum of freq (after sorting by highlow) 
    to determine which topics to include"""
    if hl_flag == 1:
        class_num = 2
        total_freq = np.sum(df_tpc.FreqRaw[(df_tpc.Class == class_num) 
                                           & (df_tpc.Topic > -1)])
        df_a = df_tpc[(df_tpc.Class==class_num) & (df_tpc.Topic > -1)]
        df_a = df_a.sort_values("HighOverLow", ascending = False)
    else:
        class_num = 1
        total_freq = np.sum(df_tpc.FreqRaw[(df_tpc.Class == class_num) 
                                           & (df_tpc.Topic > -1)])
        df_a = df_tpc[(df_tpc.Class== class_num) & (df_tpc.Topic > -1)]
        df_a = df_a.sort_values("HighOverLow", ascending = True)
        
    df_a["FreqSplit"] = df_a.apply(lambda x:
                                  (x.FreqRaw / total_freq) * 100,
                                  axis = 1)
    df_a["CumSum"] = df_a.FreqSplit.cumsum(axis = 0)
    df_a["PBTLabel"] = df_a.apply(lambda x: 
                             0 if x.CumSum < perc_best_topic else 1,
                             axis = 1)
    
    #---Get topics and indices for all but last topic
    df_di["Class"] = classes
    df_topics = df_a[df_a.PBTLabel == 0]
    topics = df_topics.Topic
    
    # if proportion of first topic doesn't exceed perc_best_topic
    if len(df_topics) > 0:    
        t_i = df_di[(df_di.Topic.isin(topics)) & 
                    (df_di.Class == class_num)].index
        last_topic = df_a.Topic.iloc[len(df_topics)]
        best_topics = np.concatenate((np.array(topics).reshape(1,-1),
                                      last_topic.reshape(1,-1)), 
                                      axis = 0).reshape(1,-1)[0]
    else:     
        t_i = np.empty(0, dtype = "int")
        last_topic = df_a.Topic.iloc[0]
        best_topics = last_topic.reshape(-1,1)[0]
        
    """For the last topic get n highest probability samples, where
    n is the remaining samples needed to make up the exact percentage of samples
    required"""
    # Get indices of last_topic
    last_n_samples = round(((perc_best_topic - 
                             np.sum((np.array(df_topics.FreqSplit)))) / 100)
                           * total_freq)
    df_lt = df_di[(df_di.Topic == last_topic) & 
                   (df_di.Class == class_num)]
    l_i = df_lt.sort_values("Probability", 
                              ascending = False).index[:last_n_samples]
    
    # Remaining indices of last topic
    remain_i = df_lt.sort_values("Probability", 
                              ascending = False).index[last_n_samples:]
    
    # Concatenate and sort indices
    pbt_i = np.sort(np.concatenate((t_i, l_i)))       
    return pbt_i, remain_i, best_topics

#--- MAIN

# Make output dir
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)   

#--- Loading
# Load y (RevHiLoRating from docs)
docs = load_pickled_df(doc_path) 
y = np.array(docs.RevHiLoRating)
# del docs  

# df_out = pd.DataFrame(columns = ["RepName", "TrSplit", 
#                                  "Estimator","TrScore", "TeScore", 
#                                  "BestEst","CVResults"])
        
# Load df_tpc
df_tpc_pickle = df_tpc_path % (topic_model_name.split("All_LX_Reviews_")[1],
                                                                   tr_split)
df_tpc = load_pickled_df(df_tpc_pickle) 

# load df_di
df_di_pickle = df_di_path % (topic_model_name, tr_split)
with open(df_di_pickle,"rb") as f:
    _ = pickle.load(f)
    df_di = pickle.load(f)             

# Load feats
X = np.load(feat_path % rep_model)

if feats_to_keep != "All":
    X = X[:,feats_to_keep]
       
#--- Slice train and test

#Split y by train and test indices
tr_i, te_i = strat_split_by_rating(doc_path,tr_split)

# Get perc_best_topic_indices
pbt_i_hi, remain_i_hi, best_topics_hi = get_perc_best_topic_idx(df_tpc,
                                                      df_di, 
                                                      perc_best_topic, 
                                                      y[tr_i], 
                                                      1)
pbt_i_lo, remain_i_lo, best_topics_lo = get_perc_best_topic_idx(df_tpc,
                                                      df_di, 
                                                      perc_best_topic, 
                                                      y[tr_i], 
                                                      0)   

"""Concatenate indices for hi and low perc_best_topic, as well as 
"remaining" indices from the last_label (to hold out later)"""
pbt_i = np.concatenate((pbt_i_hi, pbt_i_lo))
remain_i = np.concatenate((remain_i_hi, remain_i_lo))

# Get all indices not in pbt_i or remain_i to draw permutations from
perm_i_pool = np.setdiff1d(np.setdiff1d(np.arange(len(tr_i)), remain_i), 
                                       pbt_i)

# Generate list of permutations, prepend pbt_i as the first
rng = np.random.default_rng(42)
perm_list = [rng.permutation(perm_i_pool)[:len(pbt_i)] 
             for i in np.arange(n_perm)]
perm_list = [pbt_i] + perm_list

for p_i, p in enumerate(perm_list):

    #--- Slicing
    """ Slice train and test indices 
    now X_train indices are compatible with perm_list indices
    """
    X_train = X[tr_i,:]
    X_test = X[te_i,:]
    y_train = y[tr_i]
    y_test = y[te_i]
    
    # Remove perm indices from training
    X_train = np.delete(X_train,p,axis = 0)
    y_train = np.delete(y_train,p)
    
    #--- Slice high and low labels (remove zeros)
    X_train = X_train[y_train > 0,:]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0,:]
    y_test = y_test[y_test > 0] 
    
    # DELETE STUFF HERE!!!
    del tr_i, te_i, X        
   

# FIX BELOW!!!!

# Does perm need to be based on retaining equal proportions for the 2 classes?

   
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
    new_row = [m_name, ts, e_name,
               train_score, test_score,
               grid.best_estimator_,
               grid.cv_results_]
    df_out.loc[len(df_out)] = new_row

    print("Model: %s tr_split_%i %s" 
          % (m_name,ts,e_name))
    print(("Time elapsed: %1.1f, best train score: %1.4f, " +
          "best test score: %1.4f") 
          % (time.time() - tic, train_score, test_score))
                    
#Pickle
out_name = "GridMultiModel_Class_%s.pickle" % (str(time.time()).replace(".","_"))
pickle_path = os.path.join(output_path,out_name)
with open(pickle_path,"wb") as f:
    pickle.dump(df_out, f)

# Plot
x_labels = ["%s %s TS: %i" % 
            (df_out.loc[i].RepName,
             df_out.loc[i].Estimator,
            df_out.loc[i].TrSplit)            
            for i in np.arange(len(df_out))]

fig, ax = plt.subplots(figsize=(8,5), dpi = 300)

ax.plot(np.arange(len(df_out)), "TrScore", data = df_out, c = "blue")
ax.plot(np.arange(len(df_out)), "TeScore", data = df_out, c = "red")

ax.set_xticks(np.arange(len(df_out)),
                        labels = x_labels, rotation = 90)
ax.set_ylabel(mod_eval_metric)
ax.legend(labels = ["Train", "Test"])

