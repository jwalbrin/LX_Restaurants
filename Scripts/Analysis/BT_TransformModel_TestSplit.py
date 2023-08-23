# Transform held-out data with fitted model with BERTopic

import os
# import time
# import pickle
import numpy as np
from bertopic import BERTopic
import sys
scripts_path = "/home/jon/GitRepos/LX_Restaurants/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *

# import openai
# from bertopic import BERTopic
# from bertopic.representation import KeyBERTInspired
# from bertopic.representation import PartOfSpeech
# from bertopic.representation import OpenAI
# from bertopic.representation import TextGeneration
# from transformers import pipeline
# from sklearn.feature_extraction.text import CountVectorizer

output_path = "/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/"
doc_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/" + 
            "Review_Data.pickle")
tm_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" +
           "All_LX_Reviews_standard_all-MiniLM-L6-v2_Train_50")

embed_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" +
              "Embeddings/All_LX_Review_Embeddings_all-MiniLM-L6-v2.npy")

#--- MAIN

# Make out_path
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)
    
# Get strat split training indices
tr_split = int(tm_path.split("Train_")[-1])
if tr_split == 75:
    tr_i, te_i = strat_split_by_rating_75(doc_path)
elif tr_split == 50:
    tr_i, te_i = strat_split_by_rating_50(doc_path)

# Load docs, embeddings
docs = load_pickled_df(doc_path)
docs = list(docs.RevText)
embeddings = np.load(embed_path)

# Slice test indices
docs = np.array(docs)[te_i].tolist()
embeddings = embeddings[te_i]

# Load topic model
topic_model = BERTopic.load(os.path.join(tm_path))

# Transform
te_tm = topic_model.transform(docs, embeddings)

# Save topic-wise probabilities for test set
save_name = tm_path.split("/")[-1].split("Train")[0]
all_prob_mat = te_tm[1]
apm_path = os.path.join(output_path, "%sTest_%i_ProbMat" % 
                           (save_name, 100 - tr_split))
np.save(apm_path, all_prob_mat)

best_prob = te_tm[0]
bp_path = os.path.join(output_path, "%sTest_%i_BestProbVec" % 
                           (save_name, 100 - tr_split))
np.save(bp_path, best_prob)

