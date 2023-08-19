# Topic by class analyses
# Loads topic model and visualizes frequencies for each rating

import os
import time
import pickle
import numpy as np
from bertopic import BERTopic
import sys
scripts_path = "/home/jon/GitRepos/LX_Restaurants/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import plotly.io as io
io.renderers.default='browser'

# output_path = "/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/"
# output_name_stem = "All_LX_Reviews"
doc_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/" + 
            "Review_Data.pickle")
tm_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" +
              "All_LX_Reviews_standard_all-MiniLM-L6-v2_Train_50")

# tm_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" +
#               "All_LX_Reviews_chatgpt_all-MiniLM-L6-v2_Train_50")

#--- MAIN

#--- Load

# Get strat split training indices
tr_split = int(tm_path.split("_")[-1])
if tr_split == 75:
    tr_i, _ = strat_split_by_rating_75(doc_path)
elif tr_split == 50:
    tr_i, _ = strat_split_by_rating_50(doc_path)
        
# Load review data, get classes, docs as a list
docs = load_pickled_df(doc_path)
classes = ["%i star" % i for i in docs.RevRating]
classes = np.array(classes)[tr_i].tolist()
class_counts = docs.RevRating.value_counts()
docs = list(docs.RevText)
docs = np.array(docs)[tr_i].tolist()
del tr_i

# Load topic model
topic_model = BERTopic.load(os.path.join(tm_path))

# Manually create topics pc; MORE SEUCRE WAY OF JOINING CLASSES!!!
df_di = topic_model.get_document_info(docs)
df_di["Class"] = classes

# Get Topic, Frequency, Class
new_tpc = df_di[["Topic", "Document", "Class"]].copy()
new_tpc = new_tpc.groupby(["Class", "Topic"], as_index=False).count()
new_tpc.rename(columns = {"Document": "Frequency"}, inplace = True)
# new_tpc = new_tpc.sort_values(["Class","Topic"], ascending = True)
# new_tpc = new_tpc.join(df_di[["Topic","Name"]], on = "Topic", lsuffix= "", rsuffix = "_right")

# Unique names, sorted
u_topic_names = df_di[["Topic", "Name"]].sort_values("Topic")
u_topic_names = u_topic_names.drop_duplicates()

# Merge the two
new_tpc = new_tpc.merge(u_topic_names, how = "outer")

new_tpc["Frequency"] = new_tpc.apply(lambda x: x.Frequency / 
                               class_counts[int(x.Class[0].split(" ")[0])], 
                               axis = 1)
new_tpc = new_tpc.sort_values("Class",ascending = False).reset_index(drop = True)

topics_pc = new_tpc

# new_tpc = new_tpc.join(df_di[["Topic","Name"]], on = "Topic",
#                        lsuffix= "_left", rsuffix = "_right")

# a = df_di[["Topic","Name","Document"]]
# a.set_index("Topic", inplace = True)
# b = new_tpc[["Topic", "Frequency","Class"]]
# b.set_index("Topic", inplace = True)

# c = b.join(a, how = "outer")

# d = pd.merge(a, b, left_index = True, right_index = True, 
#              how = 'outer')

# c = b.merge(a, on = "Topic", validate = "m:1")

# a = new_tpc.join(df_di[["Topic","Name"]], on = "Topic",
#                         lsuffix= "_left", rsuffix = "_right")
# merged_df = pd.merge(new_tpc, df_di, left_on='Topic', right_on='Topic')

# # new_tpc = a.groupby(["Class", "Topic"]).count()
# # new_tpc = new_tpc.sort_values(["Class","Topic"], ascending = True)

# b = topics_pc.sort_values(["Class","Topic"], ascending = True)

# c = np.concatenate((new_tpc,b),axis = 1)

#--- Topics per class
# # Run
# topics_pc = topic_model.topics_per_class(docs, classes=classes)

# # Scale topics by counts
# topics_pc["Frequency"] = topics_pc.apply(lambda x: x.Frequency / 
#                                 class_counts[int(x.Class[0].split(" ")[0])], 
#                                 axis = 1)
# topics_pc = topics_pc.sort_values("Class", ascending = False)

# #--- Visualize (built-in)
# topic_model.visualize_topics_per_class(topics_pc, top_n_topics=30)

#--- Bar plot per rating (x = topics)
# Aesthetics
cmap = get_cmap("viridis")
y_lims = [0,0.1]
y_label = "Prop. of Reviews"
cm = 1/2.54

# Loop each set of ratings
u_ratings = list(set(classes))
u_ratings.sort(reverse = True)
top_n_to_show = 10
for r in u_ratings:

    fig, ax = plt.subplots(figsize=(17*cm, 6*cm), dpi = 300)
    
    plot_data = topics_pc[(topics_pc.Class == r) & 
                          (topics_pc.Topic != -1)]
    plot_data = plot_data.nlargest(top_n_to_show, "Frequency")
    
    x_labels = [("%i %s" % (plot_data.Topic.iloc[i], plot_data.Name.iloc[i]))
                for i in np.arange(len(plot_data))]
    
    cmap_intervals = np.linspace(0,1,len(x_labels))
    
    ax.set_ylabel(y_label)
    ax.set_xlabel("Topic")
    ax.set_ylim(y_lims)
    ax.title.set_text(r + " ratings")
    ax.set_xticks(np.arange(0,len(plot_data)),
                  labels = x_labels, rotation = 90)
    
    ax.bar(x = np.arange(len(plot_data)), height = "Frequency", 
           data = plot_data,
           color = cmap(cmap_intervals))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

#--- Bar plot per topic (X = ratings)
# topics_to_show = np.array([0,1])
topics_to_show = np.arange(20)
topics_to_show = np.arange(11)


for t in topics_to_show:
    fig, ax = plt.subplots(figsize=(8*cm, 6*cm), dpi = 300)    
    plot_data = topics_pc[topics_pc.Topic == t]
    
    # Add missing values
    if len(plot_data) < len(u_ratings):
        miss_ratings = [i for i in u_ratings 
                       if i not in list(plot_data.Class)]        
        for mr in np.arange(len(miss_ratings)):        
            new_row = [t, "_", 0, miss_ratings[mr], "_"]
            plot_data.loc[len(plot_data)] = new_row
    
    x_labels = list(plot_data.Class)
    x_labels = [i[0] for i in x_labels]
    
    cmap_intervals = np.linspace(0,1,len(x_labels))
    
    ax.set_ylabel(y_label)
    ax.set_xlabel("Star Rating")
    ax.set_ylim(y_lims)
    ax.title.set_text("Topic %i %s" % (t, plot_data.Words.iloc[0]))
    ax.set_xticks(np.arange(0,len(plot_data)),
                  labels = x_labels, rotation = 0)
    
    ax.bar(x = np.arange(len(plot_data)), height = "Frequency", 
           data = plot_data,
           color = cmap(cmap_intervals))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
