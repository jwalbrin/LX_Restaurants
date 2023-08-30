# Create embeddings for reviews for later use
# Applies UMap too

import os
import sys
import numpy as np
from umap import UMAP
from sentence_transformers import SentenceTransformer   

scripts_path = "/home/jon/GitRepos/LX_Restaurants/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *

output_path = "/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/Embeddings/"
output_name = "All_LX_Review_Embeddings"
data_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/")
embed_model = "all-MiniLM-L6-v2"

# default
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

umap_model = UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine')

umap_model = UMAP(n_neighbors=15, n_components=50, min_dist=0.0, metric='cosine')

#--- MAIN

# Make out_path
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)

# Load documents
docs = load_pickled_df(data_path + "Review_Data.pickle")
docs = list(docs.RevText)

# Prepare embeddings
sentence_model = SentenceTransformer(embed_model)
embeddings = sentence_model.encode(docs, show_progress_bar=True)

# umap-reduced embeddings
embeddings = umap_model.fit_transform(embeddings)

# Save
np.save(os.path.join(output_path,"%s_%s_UMAP_%i" %
                     (output_name,embed_model,
                      umap_model.get_params()["n_components"])), 
                      embeddings)

