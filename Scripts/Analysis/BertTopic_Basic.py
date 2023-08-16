
import os
import time
from bertopic import BERTopic
import sys
scripts_path = "/home/jon/GitRepos/LX_Restaurants/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *

import plotly.io as io
io.renderers.default='browser'


output_path = "/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/"
output_name = "All_LX_Reviews"
data_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/")


# input list delete
# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

#--- MAIN

# Make out_path
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)
    
docs = load_pickled_df(data_path + "Review_Data.pickle")
docs = list(docs.RevText)

#--- Create model
tic = time.time()
topic_model = BERTopic()
# topic_model.get_params() # show params
topics, probs = topic_model.fit_transform(docs)
print("Run time: %1.1f seconds" % (time.time() - tic))

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
# topic_model.visualize_barchart(top_n_topics=100)

