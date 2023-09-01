# Reduce clusters of pre-fitted model

import os
import time
import pickle
import numpy as np
from bertopic import BERTopic
import sys
scripts_path = "/home/jon/GitRepos/LX_Restaurants/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *

import openai
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import PartOfSpeech
from bertopic.representation import OpenAI
from bertopic.representation import TextGeneration
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer

import plotly.io as io
io.renderers.default='browser'

output_path = "/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/"
doc_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/" + 
            "Review_Data.pickle")

tm_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" +
              "All_LX_Reviews_standard_all-MiniLM-L6-v2_Train_75")

tr_split = 75
reduc_mod_sizes = [90, 80, 70, 60]

#--- MAIN

# Make out_path
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)
    
# Get strat split training indices
tr_i, te_i = strat_split_by_rating(doc_path,tr_split)

# Load docs, embeddings
docs = load_pickled_df(doc_path)
docs = list(docs.RevText)

# Slice training indices
docs = np.array(docs)[tr_i].tolist()

# Load topic model
topic_model = BERTopic.load(os.path.join(tm_path))

# Out name
out_name = tm_path.split("/")[-1]

#--- Reduce model
for rm in reduc_mod_sizes:
    # Run reduced model
    reduc_mod = topic_model.reduce_topics(docs, nr_topics=rm)
    
    # Save model 
    reduc_mod.save(os.path.join(output_path,
                                  "%s_Reduc_%i_Clusters" % 
                                  (out_name,rm)))

    # Save info
    df_ti = reduc_mod.get_topic_info()
    df_di = reduc_mod.get_document_info(docs)
    pickle_path = os.path.join(output_path, 
                               "%s_Reduc_%i_Clusters_Info.pickle" % 
                               (out_name,rm))
    with open(pickle_path,"wb") as f:
        pickle.dump(df_ti, f)
        pickle.dump(df_di, f)  
        
    # Save Topic-wise probabilities
    all_prob_mat = reduc_mod.probabilities_
    apm_path = os.path.join(output_path, 
                               "%s_Reduc_%i_Clusters_ProbMat" % 
                               (out_name,rm))
    np.save(apm_path, all_prob_mat)


 





