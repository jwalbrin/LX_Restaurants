"""Uses grid search to find best classification model, 
for different input topic rep_models
Version C is binary HiLow classification
P = probmat embeddings
Perm = permutations where a specified percentage of samples from each
class are removed (e.g. perc_best_topics = 10 = 10%): for the first "test"
permutation, these indices are the hypothesized "best samples" that belong to 
the topics that best separate high and low ratings; the remaining permuatations 
are indices of the same % (e.g. 10% samples per class) that are not included 
in these "best samples"
Note that in most cases, a small set of samples are excluded from all analyses.
For example if perc_best_topics = 10%, and the first 4 topics account for 10.3%
of samples, then 0.3% remaining samples from the 4th topic will be excluded 
from all permutations.
"""

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

feat_path = (main_path + "%s_%s_ProbMat.npy")
# feat_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/Embeddings/" +
#              "%s.npy")

doc_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/" +
            "Review_Data.pickle")

df_tpc_path = (main_path + 
               "ByClass/All_LX_Reviews_ByClass_%s_Train_%i.pickle")

df_di_path = (main_path + 
               "%s_Train_%i_Info.pickle")

output_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Classification/Perm/")

# embed_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" +
#               "Embeddings/All_LX_Review_Embeddings_all-MiniLM-L6-v2.npy")

# Test parameters
# rep_model = "All_LX_Review_Embeddings_all-MiniLM-L6-v2_UMAP_5"
# rep_model = "All_LX_Review_Embeddings_all-MiniLM-L6-v2"

topic_model_name = "All_LX_Reviews_keybert_all-MiniLM-L6-v2"

tr_split = 75

feats_to_keep = np.arange(50) # "All" or np.array of indices
feats_to_keep = "All"

mod_eval_metric = "balanced_accuracy"

cv_folds = 10

n_perm = 50

perc_best_topic = 20 # percentage of best topics to remove


# pipe

pipe = Pipeline([
              ('scaler', StandardScaler()),
              ('clf', LogisticRegression(max_iter = 300, 
                                         random_state = 42))
              ])

#--- MAIN

#--- Functions 
    
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
        
        if len(df_topics) == 1:
            best_topics = np.concatenate((np.array(topics).reshape(1,-1),
                                          last_topic.reshape(1,-1)), 
                                          axis = 0).reshape(1,-1)[0]
        else:
            best_topics = np.concatenate((np.array(topics).reshape(-1,1),
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
    pbt_remain_i = df_lt.sort_values("Probability", 
                              ascending = False).index[last_n_samples:]
    
    # Concatenate and sort indices
    pbt_i = np.sort(np.concatenate((t_i, l_i)))  

    # Get other indices (all remaining indices for given class)      
    class_remain_i = df_di[(df_di.Class == class_num) & 
                     (df_di.Topic > -1) & 
                     (~df_di.Topic.isin(best_topics))].index
    
    # Raise exception if 3 sets of indices do not equal total_freq
    if len(pbt_i) + len(pbt_remain_i) + len(class_remain_i) != total_freq:
        raise Exception("# indices and total_freq do not match")
    
    return pbt_i, pbt_remain_i, class_remain_i, best_topics

#--- MAIN

# Make output dir
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)   

#--- Loading
# Load y (RevHiLoRating from docs)
docs = load_pickled_df(doc_path) 
y = np.array(docs.RevHiLoRating)
# del docs  
       
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
X_tr_main = np.load(feat_path % (topic_model_name, 
                                ("Train_%i" % 
                                 tr_split))).astype("float32")
X_te_main = np.load(feat_path % (topic_model_name, 
                                ("Test_%i" % 
                                (100 - tr_split)))).astype("float32")

# Slice feats to keep
if feats_to_keep != "All":
    X_tr_main = X_tr_main[:,feats_to_keep]
    X_te_main = X_te_main[:,feats_to_keep]
    
#--- Slice train and test

# Get train and test indices
tr_i, te_i = strat_split_by_rating(doc_path,tr_split)

# Get perc_best_topic_indices
(pbt_i_hi, 
 pbt_remain_i_hi, 
 class_remain_i_hi, 
 best_topics_hi) = get_perc_best_topic_idx(df_tpc,
                                           df_di, 
                                           perc_best_topic, 
                                           y[tr_i], 
                                           1)
(pbt_i_lo, 
 pbt_remain_i_lo, 
 class_remain_i_lo, 
 best_topics_lo) = get_perc_best_topic_idx(df_tpc,
                                           df_di, 
                                           perc_best_topic, 
                                           y[tr_i], 
                                           0)   

"""Concatenate indices for hi and low perc_best_topic, as well as 
"remaining" indices from the last_label (to hold out later)"""
pbt_i = np.concatenate((pbt_i_hi, pbt_i_lo))

"""Generate list of permutations class balanced permutations
e.g. if perc_best_topic = 10%, then 10% of samples from each class
that are not in pbt_i hi or lo
Prepend perm_list with pbt_i"""
rng = np.random.default_rng(42)
perm_list = [np.concatenate((
             rng.permutation(class_remain_i_hi)[:len(pbt_i_hi)], 
             rng.permutation(class_remain_i_lo)[:len(pbt_i_lo)]))             
             for i in np.arange(n_perm)]
perm_list = [pbt_i] + perm_list

#--- Permutations
df_out = pd.DataFrame(columns = ["TrainScore", "TestScore"])
for p_i, p in enumerate(perm_list):
    tic = time.time()

    #--- Slicing
    """ Slice train and test indices, 
    now X_train indices are compatible with perm_list indices
    """
    X_train = np.copy(X_tr_main)
    X_test = np.copy(X_te_main)
    y_train = y[tr_i]
    y_test = y[te_i]
    
    # Remove perm indices from training X,y
    X_train = np.delete(X_train,p,axis = 0)
    y_train = np.delete(y_train,p)
    
    #--- Slice high and low labels (remove zeros)
    X_train = X_train[y_train > 0,:]
    y_train = y_train[y_train > 0]
    X_test = X_test[y_test > 0,:]
    y_test = y_test[y_test > 0] 
    
    # Strat k fold (outer CV for train data)
    skf = StratifiedKFold(n_splits= cv_folds, 
                          random_state = 42, 
                          shuffle = True)
    
    # Classifier, get scores
    pipe.fit(X_train, y_train) 
    # train_score = pipe.score(X_train, y_train)
    # test_score = pipe.score(X_test, y_test)    
    
    # 
    train_score = balanced_accuracy_score(y_train, pipe.predict(X_train))
    test_score = balanced_accuracy_score(y_test, pipe.predict(X_test))
    # a = multilabel_confusion_matrix(y_test, pipe.predict(X_test))
    
    # Assign
    df_out.loc[len(df_out)] = [train_score, test_score]
    
    if p_i % 10 == 0:     
        print(("Perm %i time elapsed: %1.1f, train score: %1.4f, " +
              "test score: %1.4f") 
              % (p_i, time.time() - tic, train_score, test_score))
#Pickle
out_name = "Perm_HiLo_PBT_%i_%s.pickle" % (perc_best_topic, 
                                           str(time.time()).replace(".","_"))
pickle_path = os.path.join(output_path,out_name)
with open(pickle_path,"wb") as f:
    pickle.dump(df_out, f)
    
# plot

#--- Calculate and assign stats
C = np.sum(df_out.TestScore[1:] >= df_out.TestScore[0])
p_val = 1 - (C + 1) / (n_perm + 1)    

title = ("%s %s: %i%% sample removal (%i perms)" %
        (topic_model_name.split("All_LX_Reviews_")[1].split("_")[0],
        topic_model_name.split("All_LX_Reviews_")[1],
        perc_best_topic,
        n_perm))

fig, ax = plt.subplots()
ax.hist(df_out.TestScore.loc[1:], bins = 50)
ax.axvline(df_out.TestScore[0], color = "r")
ax.set_xlabel("Perm. Score")
ax.set_ylabel("Count")
ax.text(df_out.TestScore[0],ax.get_ylim()[1] - 5,"%1.4f" % p_val)
ax.set_title(title)

 