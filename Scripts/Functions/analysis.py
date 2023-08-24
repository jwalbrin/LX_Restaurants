# import os
import time
# import pickle
# import numpy as np
# import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import openai
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import OpenAI
from bertopic.representation import TextGeneration
from transformers import pipeline

def std_model(docs, embeddings, output_name_stem):
    """ Standard model with frequent words removed """
    tic = time.time()
    vectorizer_model = CountVectorizer(stop_words="english")

    topic_model = BERTopic(calculate_probabilities= True, 
                           vectorizer_model=vectorizer_model,
                           embedding_model= "all-MiniLM-L6-v2").fit(
                               docs,
                               embeddings)
                              
    print("Run time: %1.1f seconds" % (time.time() - tic))
    save_name = output_name_stem + "_standard"
    return topic_model, save_name

def non_std_model(docs, embeddings, model_name, output_name_stem):
    """ Select one of 4 hard-coded models """
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
    topic_model = BERTopic(calculate_probabilities= True, 
                  representation_model= representation_model, 
                  embedding_model= "all-MiniLM-L6-v2").fit(
                      docs,
                      embeddings)
                      
    print("Run time: %1.1f seconds" % (time.time() - tic))
    save_name = output_name_stem + "_" + model_name
    return topic_model, save_name

