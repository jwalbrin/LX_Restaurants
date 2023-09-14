# Fit model with BERTopic

import os
import time
import pickle
import numpy as np
from bertopic import BERTopic
import sys
scripts_path = "/home/jon/GitRepos/LX_Restaurants/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *
from Functions.analysis import *

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

output_path = "/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/NonStoch/"
output_name_stem = "All_LX_Reviews"
doc_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/" + 
            "Review_Data.pickle")
embed_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" +
              "Embeddings/All_LX_Review_Embeddings_all-MiniLM-L6-v2.npy")

model_names = ["keybert"] # "standard", "chatgpt", "flan-t5","gpt2", "keybert"
model_names = ["chatgpt"] # "standard", "chatgpt", "flan-t5","gpt2", "keybert"

tr_splits = [75] # 75, 50

trans_test = 1

rc_vals = [0] # zero skips, else take k clusters

#--- MAIN

# Make out_path
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)
for m in model_names:
    for ts in tr_splits:
    
        # Train / test splits
        tr_i, te_i = strat_split_by_rating(doc_path, ts)
        
        # Load docs, embeddings
        docs = load_pickled_df(doc_path)
        docs = list(docs.RevText)
        embeddings = np.load(embed_path)
        
        # # Slice training/testing indices
        # docs_tr = np.array(docs)[tr_i].tolist()
        # embeddings_tr = embeddings[tr_i]
        # docs_te = np.array(docs)[te_i].tolist()
        # embeddings_te = embeddings[te_i]
                
        #--- Run model (with training data)
        if m == "standard":
            topic_model, save_name = std_model(np.array(docs)[tr_i].tolist(),
                                               embeddings[tr_i],
                                               output_name_stem)
        else:
            topic_model, save_name = non_std_model(np.array(docs)[tr_i].tolist(),
                                                   embeddings[tr_i], 
                                                   m,
                                                   output_name_stem)  
           
        # Save model 
        embed_name = embed_path.split("_")[-1].split(".npy")[0]
        topic_model.save(os.path.join(output_path,
                                      "%s_%s_Train_%i" % 
                                      (save_name,embed_name, ts)))
        
        # Save info
        df_ti = topic_model.get_topic_info()
        df_di = topic_model.get_document_info(np.array(docs)[tr_i].tolist())
        pickle_path = os.path.join(output_path, "%s_%s_Train_%s_Info.pickle" % 
                                   (save_name, embed_name, ts))
        with open(pickle_path,"wb") as f:
            pickle.dump(df_ti, f)
            pickle.dump(df_di, f)  
            
        # Save Topic-wise probabilities
        all_prob_mat = topic_model.probabilities_
        apm_path = os.path.join(output_path, "%s_%s_Train_%s_ProbMat" % 
                                   (save_name, embed_name, ts))
        np.save(apm_path, all_prob_mat)
        
        #--- Transform test data
        if trans_test == 1:
            
            # Transform test data            
            te_tm = topic_model.transform(np.array(docs)[te_i].tolist(),
                                          embeddings[te_i])
            
            # Save Topic-wise probabilities for test set
            all_prob_mat = te_tm[1]
            apm_path = os.path.join(output_path, "%s_%s_Test_%s_ProbMat" % 
                                       (save_name, embed_name, 100 - ts))
            np.save(apm_path, all_prob_mat)
            
            best_prob = te_tm[0]
            bp_path = os.path.join(output_path, "%s_%s_Test_%s_BestProbVec" % 
                                       (save_name, embed_name, 100 - ts))
            np.save(bp_path, best_prob)
                    
        #--- Reduce clusters
        for rc in rc_vals:
            
            if rc != 0:
                
                # Re-load topic model (non-reduced version)
                # Load topic model
                topic_model = BERTopic.load(os.path.join(output_path,
                                              "%s_%s_Train_%i" % 
                                              (save_name,embed_name, ts)))
                
                # Get reduced model (for training topics)
                reduc_mod = topic_model.reduce_topics(
                                        np.array(docs)[tr_i].tolist(), 
                                            nr_topics=rc)
                
                # Save model 
                reduc_mod.save(os.path.join(output_path,
                                        "%s_%s_Train_%i_Reduc_%i_Clusters" % 
                                        (save_name,embed_name, 
                                         ts,rc)))               
                                            
                # Save info
                df_ti = reduc_mod.get_topic_info()
                df_di = reduc_mod.get_document_info(np.array(docs)[tr_i].tolist())
                pickle_path = os.path.join(output_path, 
                               "%s_%s_Train_%i_Reduc_%i_Clusters_Info.pickle" % 
                               (save_name,embed_name, 
                                ts,rc))
                with open(pickle_path,"wb") as f:
                    pickle.dump(df_ti, f)
                    pickle.dump(df_di, f)  
                    
                # Save Topic-wise probabilities
                all_prob_mat = reduc_mod.probabilities_
                apm_path = os.path.join(output_path, 
                                "%s_%s_Train_%i_Reduc_%i_Clusters_ProbMat" % 
                                (save_name,embed_name, 
                                 ts,rc))
                np.save(apm_path, all_prob_mat)              
                                           
                #--- Transform test data
                if trans_test == 1:
                    
                    # Transform test data            
                    te_tm = reduc_mod.transform(np.array(docs)[te_i].tolist(),
                                                  embeddings[te_i])
                    
                    # Save Topic-wise probabilities for test set
                    all_prob_mat = te_tm[1]
                    apm_path = os.path.join(output_path, 
                                    "%s_%s_Test_%s_Reduc_%i_Clusters_ProbMat" % 
                                    (save_name, embed_name, 
                                     100 - ts, rc))
                    np.save(apm_path, all_prob_mat)
                    
                    best_prob = te_tm[0]
                    bp_path = os.path.join(output_path, 
                                    "%s_%s_Test_%s_Reduc_%i_Clusters_BestProbVec" % 
                                    (save_name, embed_name, 
                                     100 - ts, rc))
                    np.save(bp_path, best_prob)                               
                    
                    # # Transform test data            
                    # te_tm = topic_model.transform(np.array(docs)[te_i].tolist(),
                    #                               embeddings[te_i])
                    
                    # # Save Topic-wise probabilities for test set
                    # all_prob_mat = te_tm[1]
                    # apm_path = os.path.join(output_path, "%s_%s_Test_%s_ProbMat" % 
                    #                            (save_name, embed_name, 100 - tr_split))
                    # np.save(apm_path, all_prob_mat)
                    
                    # best_prob = te_tm[0]
                    # bp_path = os.path.join(output_path, "%s_%s_Test_%s_BestProbVec" % 
                    #                            (save_name, embed_name, 100 - tr_split))
                    # np.save(bp_path, best_prob)
            






