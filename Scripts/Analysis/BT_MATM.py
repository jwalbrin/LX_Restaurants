"""Multi-aspect topic modelling for:
KeyBERTInspired
PartOfSpeech
FLAN-T5
ChatGPT (labels)    
"""    
    
import os
import sys
import time
import pickle
import openai
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import PartOfSpeech
from bertopic.representation import OpenAI
from bertopic.representation import TextGeneration
from transformers import pipeline

scripts_path = "/home/jon/GitRepos/LX_Restaurants/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *

import plotly.io as io
io.renderers.default='browser'


output_path = "/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/"
output_name = "All_LX_Reviews_MATM_TEST"
data_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/")


# input list delete
# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

#--- MAIN

# Make out_path
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)
    
docs = load_pickled_df(data_path + "Review_Data.pickle")
docs = list(docs.RevText)
docs = docs[:15_000]

#--- Create model

# The main representation of a topic
main_representation = KeyBERTInspired()

# Model 2 (Chat GPT)
openai.api_key = "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"
prompt = """
I have a topic that contains the following documents: 
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short topic label in the following format:
topic: <topic label>
"""
aspect_model1 = OpenAI(model="gpt-3.5-turbo",
                              prompt = prompt,
                              delay_in_seconds=10, chat=True)

# Model 3 (GPT2)
aspect_model2 = TextGeneration('gpt2')

# Model 4 (FLAN-T5)
prompt = ("I have a topic described by the following keywords: [KEYWORDS]. " +
          "Based on the previous keywords, what is this topic about?")

# Create your representation model
generator = pipeline('text2text-generation', model='google/flan-t5-base')
aspect_model3 = TextGeneration(generator, prompt=prompt)

# Add all models together to be run in a single `fit`
representation_model = {
   "Main": main_representation,
   "Aspect1":  aspect_model1,
   "Aspect2":  aspect_model2,
   "Aspect3":  aspect_model3,
}

tic = time.time()
topic_model = BERTopic(representation_model=representation_model, 
                       embedding_model= "all-MiniLM-L6-v2").fit(docs)
print("Run time: %1.1f seconds" % (time.time() - tic))

# tic = time.time()
# # topic_model.get_params() # show params
# topics, probs = topic_model.fit_transform(docs)
# print("Run time: %1.1f seconds" % (time.time() - tic))

#--- Get info

# Topic info (cluster number, count, names)
df_ti = topic_model.get_topic_info()

"""
Document info
Name, Topic, Top_n_words are repeats; probabilities; represnetative doc is 
true for 1 doc, false for all else"""
df_di = topic_model.get_document_info(docs)

# For given topic get
topic_model.get_topic(0)

# Save model and info
topic_model.save(os.path.join(output_path,
      output_name))

pickle_path = os.path.join(output_path, "%s_Info.pickle" % output_name)
with open(pickle_path,"wb") as f:
    pickle.dump(df_ti, f)
    pickle.dump(df_di, f)  

# # Load model
# topic_model = BERTopic.load(os.path.join(output_path,output_name))

# # visualize
# topic_model.visualize_distribution(np.array(df_di.Probability))
# topic_model.visualize_topics()
topic_model.visualize_barchart(top_n_topics=100)

