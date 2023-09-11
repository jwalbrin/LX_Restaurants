"""Create and save topics_by_class df
Version B = hilo ratings"""

import os
import pickle
import numpy as np
from scipy.stats import linregress
from bertopic import BERTopic
import sys
scripts_path = "/home/jon/GitRepos/LX_Restaurants/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *

output_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/ByClass/" + 
               "NonStoch/")
output_name_stem = "All_LX_Reviews_ByClass"
doc_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/" + 
            "Review_Data.pickle")
tm_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" +
              "All_LX_Reviews_standard_all-MiniLM-L6-v2_Train_%s")

tm_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/NonStoch/" +
              "All_LX_Reviews_standard_all-MiniLM-L6-v2_Train_%s")


# tm_path = ("/home/jon/GitRepos/LX_Restaurants/Output/BertTopic/" +
#               "All_LX_Reviews_standard_all-MiniLM-L6-v2_Train_%s" +
#               "_Reduc_60_Clusters")

tr_split = 75


#--- MAIN

# Make out_path
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)

#--- Functions
def create_topics_per_class_df(topic_model, docs, classes, class_counts):
    """ Outputs a similar df to that based on the built-in 
    topics_per_class method
    """
    #
    # Create df_tpc
    df_di = topic_model.get_document_info(docs)
    df_di["Class"] = classes
    
    # Get Topic, Frequency, Class
    df_tpc = df_di[["Topic", "Document", "Class"]].copy()
    df_tpc = df_tpc.groupby(["Class", "Topic"], as_index=False).count()
    df_tpc.rename(columns = {"Document": "FreqTotal"}, inplace = True)
    
    # Unique topic names, sorted
    u_topic_names = df_di[["Topic", "Name"]].sort_values("Topic")
    u_topic_names = u_topic_names.drop_duplicates()
    
    # Unique class names
    u_class = list(set(classes))
    u_class.sort(reverse = True)
    
    # Merge the two (implicitly inner join, on topic)
    df_tpc = df_tpc.merge(u_topic_names)
    
    # Add missing rows (where topic did not appear for that rating)
    for u in u_class:
        if len(df_tpc[df_tpc.Class == u]) < len(u_topic_names):
            miss_topics = np.setdiff1d(u_topic_names.Topic, 
                                       df_tpc.Topic[df_tpc.Class == u])
            for mt in miss_topics:        
                new_row = [u, mt, 0,
                           u_topic_names.Name[u_topic_names.Topic == mt].iloc[0]]
                df_tpc.loc[len(df_tpc)] = new_row
            
    # Convert frequency to % (of all reviews per rating category)   
    df_tpc["FreqRaw"] = df_tpc["FreqTotal"]
    df_tpc["FreqTotal"] = df_tpc.apply(lambda x: ((x.FreqTotal / 
                                    class_counts[x.Class])
                                    *100), 
                                    axis = 1)
    df_tpc = df_tpc.sort_values(["Class", "Topic"],
                                  ascending = [False, 
                                               True]).reset_index(drop = True)
    # Reorder
    df_tpc = df_tpc[["Name", "Topic", "FreqTotal", "FreqRaw","Class"]]
    return df_tpc

def make_diff_cols(df_tpc):
    """Simple subtraction difference columns, e.g. high frequency minus 
    low frequency"""

    df_tpc["HighOverLow"] = np.zeros(len(df_tpc))
    df_tpc["HighOverMed"] = np.zeros(len(df_tpc))
    df_tpc["LowOverMed"] = np.zeros(len(df_tpc))
    n_classes = len(df_tpc[df_tpc.Topic == 0])
    
    for t_i in np.arange(len(df_tpc)):
        
        ratings = np.array(df_tpc[df_tpc.Topic == df_tpc.iloc[t_i].Topic].
                           sort_values("Class", ascending = False).
                           FreqTotal)   
            
        # # High over low
        df_tpc["HighOverLow"].iloc[t_i] = ratings[0] - ratings[1]    
        df_tpc["HighOverMed"].iloc[t_i] = ratings[0] - ratings[2]    
        df_tpc["LowOverMed"].iloc[t_i] = ratings[1] - ratings[2]    
    return df_tpc

#--- Load

# Get strat split training indices
tr_i, te_i = strat_split_by_rating(doc_path, tr_split)

# tr_split = int(tm_path.split("_")[-1])
# if tr_split == 75:
#     tr_i, _ = strat_split_by_rating_75(doc_path)
# elif tr_split == 50:
#     tr_i, _ = strat_split_by_rating_50(doc_path)
        
# Load review data, get classes, docs as a list
docs = load_pickled_df(doc_path)
classes = np.array(docs.RevHiLoRating)[tr_i].tolist()
# classes = ["%i star" % i for i in docs.RevRating]
# classes = np.array(classes)[tr_i].tolist()
class_counts = docs.RevHiLoRating.value_counts()
docs = list(docs.RevText)
docs = np.array(docs)[tr_i].tolist()
del tr_i

# Load topic model
topic_model = BERTopic.load(os.path.join(tm_path % tr_split))

# Create topics_per_class df
df_tpc = create_topics_per_class_df(topic_model, docs, classes, class_counts)


# Append diff variables
df_out = make_diff_cols(df_tpc)

# Save
save_name = tm_path.split("All_LX_Reviews_")[-1]
pickle_path = os.path.join(output_path, "%s_%s.pickle" % 
                           (output_name_stem,
                           save_name % (tr_split)))
with open(pickle_path,"wb") as f:
    pickle.dump(df_out, f)

