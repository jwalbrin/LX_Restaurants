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
output_name_stem = "All_LX_Reviews"
doc_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/" + 
            "Review_Data.pickle")
embed_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" +
              "Embeddings/All_LX_Review_Embeddings_all-MiniLM-L6-v2.npy")

model_name = "standard" # "standard", "chatgpt", "flan-t5","gpt2", "keybert"
# model_name =  "chatgpt"
# model_name =  "keybert"

tr_split = 50 # 75, 50

#--- MAIN

# Make out_path
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)
    
# Get strat split training indices
if tr_split == 75:
    tr_i, _ = strat_split_by_rating_75(doc_path)
elif tr_split == 50:
    tr_i, _ = strat_split_by_rating_50(doc_path)

# Load docs, embeddings
docs = load_pickled_df(doc_path)
docs = list(docs.RevText)
embeddings = np.load(embed_path)

# Slice training indices
docs = np.array(docs)[tr_i].tolist()
embeddings = embeddings[tr_i]

#--- Create model
def std_model(docs, embeddings):
    # Standard model with frequent words removed
    tic = time.time()
    vectorizer_model = CountVectorizer(stop_words="english")
    # topic_model = BERTopic(vectorizer_model=vectorizer_model,
    #                        embedding_model= "all-MiniLM-L6-v2").fit(docs,
    #                                                             embeddings)
    # topic_model = BERTopic(nr_topics=50, vectorizer_model=vectorizer_model,
    #                        embedding_model= "all-MiniLM-L6-v2").fit(docs,
    #                                                                 embeddings)
    topic_model = BERTopic(calculate_probabilities= True, nr_topics=50, vectorizer_model=vectorizer_model,
                           embedding_model= "all-MiniLM-L6-v2").fit(docs,
                                                                    embeddings)
                              
    print("Run time: %1.1f seconds" % (time.time() - tic))
    save_name = output_name_stem + "_standard"
    return topic_model, save_name

def non_std_model(docs, embeddings, model_name):
    #Select one of 4 hard-coded models
    if model_name == "chatgpt":
        prompt = """
        I have a topic that contains the following documents: 
        [DOCUMENTS]
        The topic is described by the following keywords: [KEYWORDS]
        
        Based on the information above, extract a short topic label in the following format:
        topic: <topic label>
        """
        openai.api_key = "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"
        representation_model = OpenAI(model="gpt-3.5-turbo",
                                      prompt = prompt,
                                      delay_in_seconds=40, chat=True)
    elif model_name == "flan-t5":
        prompt = ("I have a topic described by the following " + 
                  "keywords: [KEYWORDS]. Based on the previous keywords, " + 
                  "what is this topic about?")
        generator = pipeline('text2text-generation', 
                             model='google/flan-t5-base')
        representation_model = TextGeneration(generator, prompt=prompt)

    elif model_name == "gpt2":
        representation_model = TextGeneration('gpt2')
        
    elif model_name == "keybert":
        representation_model = KeyBERTInspired()
    
    #--- Run model
    tic = time.time()
    topic_model = BERTopic(calculate_probabilities= True, representation_model= representation_model, 
                           embedding_model= "all-MiniLM-L6-v2").fit(docs,
                                                                    embeddings)
    print("Run time: %1.1f seconds" % (time.time() - tic))
    save_name = output_name_stem + "_" + model_name
    return topic_model, save_name

if model_name == "standard":
    topic_model, save_name = std_model(docs, embeddings)
else:
    topic_model, save_name = non_std_model(docs, embeddings, model_name)
    
# Save model 
embed_name = embed_path.split("_")[-1].split(".npy")[0]
topic_model.save(os.path.join(output_path,
                              "%s_%s_Train_%i" % 
                              (save_name,embed_name, tr_split)))

# Save info
df_ti = topic_model.get_topic_info()
df_di = topic_model.get_document_info(docs)
pickle_path = os.path.join(output_path, "%s_%s_Train_%s_Info.pickle" % 
                           (save_name, embed_name, tr_split))
with open(pickle_path,"wb") as f:
    pickle.dump(df_ti, f)
    pickle.dump(df_di, f)  
    
# Save Topic-wise probabilities
all_prob_mat = topic_model.probabilities_
apm_path = os.path.join(output_path, "%s_%s_Train_%s_ProbMat" % 
                           (save_name, embed_name, tr_split))
np.save(apm_path, all_prob_mat)



    


# Load
# pickle_path = os.path.join(output_path,
#                             "All_LX_Reviews_Info_15K.pickle")
# with open(pickle_path,"rb") as f:
#     reg =  pickle.load(f)
 





