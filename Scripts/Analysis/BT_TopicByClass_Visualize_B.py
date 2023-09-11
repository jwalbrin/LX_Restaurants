# Create and save topics_by_class df

import os
import pickle
import numpy as np
from scipy.stats import linregress
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

output_path = "/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/ByClass/NonStoch/"
output_name_stem = "All_LX_Reviews_ByClass"
doc_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/" + 
            "Review_Data.pickle")

tpc_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/ByClass/NonStoch/" +
            "All_LX_Reviews_ByClass_standard_all-MiniLM-L6-v2_Train_50.pickle")
# tpc_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/ByClass/" +
#             "All_LX_Reviews_ByClass_keybert_all-MiniLM-L6-v2_Train_75_"+
#             "Reduc_100_Clusters.pickle")

# Figure aesthetics
cmap = get_cmap("viridis")
x_lims = [0,10]
cm = 1/2.54

#--- MAIN

# Load tpc
df_tpc = pd.read_pickle(tpc_path)

# remove mid ratings (0 class)
df_tpc = df_tpc[df_tpc.Class > 0]

# Unique class names
u_class = list(df_tpc.Class.unique())
u_class.sort(reverse = True)
u_class_names = ["High", "Low"]

#--- TOP N TO SHOW
#--- Bar plot per rating (x = topics)
# Loop each set of ratings
top_n_to_show = 10
for r_i, r in enumerate(u_class):
    
    fig, ax = plt.subplots(figsize=(6*cm, 6*cm), dpi = 300)
    
    plot_data = df_tpc[(df_tpc.Class == r) & 
                          (df_tpc.Topic != -1)]
    plot_data = plot_data.nlargest(top_n_to_show, "FreqTotal")
    plot_data = plot_data.sort_values("FreqTotal", ascending = True)    
   
    y_labels = [("%s" % (plot_data.Name.iloc[i]))
                for i in np.arange(len(plot_data))]    
   
    cmap_intervals = np.linspace(1,0,len(y_labels))
    
    ax.set_ylabel("Topic")
    ax.set_xlabel("Reviews (%)")

    ax.set_xlim(x_lims)
    ax.title.set_text(u_class_names[r_i] + " ratings")
    ax.set_yticks(np.arange(0,len(plot_data)),
                  labels = y_labels, rotation = 0)
    ax.set_xticks(np.arange(0,x_lims[1] + 1),
                  labels = np.arange(0,x_lims[1] + 1), rotation = 0)
    
    ax.barh(y = np.arange(len(plot_data)), width = "FreqTotal", 
            data = plot_data,
            color = cmap(cmap_intervals))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

#--- Bar plot per topic (X = ratings)
# topics_to_show = np.array([0,1])
topics_to_show = np.arange(0)
# topics_to_show = np.arange(11)
# topics_to_show = np.arange(100,110)

for t in topics_to_show:
    fig, ax = plt.subplots(figsize=(8*cm, 6*cm), dpi = 300)    
    plot_data = df_tpc[df_tpc.Topic == t]
    plot_data = plot_data.sort_values("Class", ascending = False)
    
    y_labels = u_class_names
    # y_labels = [i[0] for i in y_labels]
    
    cmap_intervals = np.linspace(1,0,len(y_labels))
    
    ax.set_xlabel("Reviews (%)")
    ax.set_ylabel("Star Rating")
    ax.set_xlim(x_lims)
    ax.title.set_text("Topic %s" % (plot_data.Name.iloc[0]))
    ax.set_yticks(np.arange(0,len(plot_data)),
                  labels = y_labels, rotation = 0)
    ax.set_xticks(np.arange(0,x_lims[1] + 1),
                  labels = np.arange(0,x_lims[1] + 1), rotation = 0)    
    
    ax.barh(y = np.arange(len(plot_data)), width = "FreqTotal", 
            data = plot_data,
            color = cmap(cmap_intervals))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
#--- TOP TOPICS BY FEATURE VARIABLES

#--- High > low
r = 2 # 2 for high, 1 for low
top_n_to_show = 10
fig, ax = plt.subplots(figsize=(6*cm, 6*cm), dpi = 300)

plot_data = df_tpc[(df_tpc.Class == r) & 
                      (df_tpc.Topic != -1)]
plot_data = plot_data.nlargest(top_n_to_show, "HighOverLow")
plot_data = plot_data.sort_values("FreqTotal", ascending = True)    
   
y_labels = [("%s" % (plot_data.Name.iloc[i]))
            for i in np.arange(len(plot_data))]    
   
cmap_intervals = np.linspace(1,0,len(y_labels))

ax.set_ylabel("Topic")
ax.set_xlabel("Reviews (%)")

ax.set_xlim(x_lims)
ax.title.set_text("High > low ratings")
ax.set_yticks(np.arange(0,len(plot_data)),
              labels = y_labels, rotation = 0)
ax.set_xticks(np.arange(0,x_lims[1] + 1),
              labels = np.arange(0,x_lims[1] + 1), rotation = 0)

ax.barh(y = np.arange(len(plot_data)), width = "FreqTotal", 
        data = plot_data,
        color = cmap(cmap_intervals))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

#--- Low > high
r = 1 # 2 for high, 1 for low
top_n_to_show = 10
fig, ax = plt.subplots(figsize=(6*cm, 6*cm), dpi = 300)

plot_data = df_tpc[(df_tpc.Class == r) & 
                      (df_tpc.Topic != -1)]
plot_data = plot_data.nsmallest(top_n_to_show, "HighOverLow")
plot_data = plot_data.sort_values("FreqTotal", ascending = True)    
   
y_labels = [("%s" % (plot_data.Name.iloc[i]))
            for i in np.arange(len(plot_data))]    
   
cmap_intervals = np.linspace(1,0,len(y_labels))

ax.set_ylabel("Topic")
ax.set_xlabel("Reviews (%)")

ax.set_xlim(x_lims)
ax.title.set_text("Low > high ratings")
ax.set_yticks(np.arange(0,len(plot_data)),
              labels = y_labels, rotation = 0)
ax.set_xticks(np.arange(0,x_lims[1] + 1),
              labels = np.arange(0,x_lims[1] + 1), rotation = 0)

ax.barh(y = np.arange(len(plot_data)), width = "FreqTotal", 
        data = plot_data,
        color = cmap(cmap_intervals))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)




