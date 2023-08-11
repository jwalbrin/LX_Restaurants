# 
"""

main - PK, Name, SkipCode, TotalReviews, 5 ratings, cuisine_FK (to cuisines PK),
       special

long_string_info - main_FK, URL, street_address, about
review - main_FK, tag, title, date, rating, text
cuisine - main_FK, 110 cols
Special diets - main_FK, 4 cols
Meals - main_FK, 6 cols
Features - main_FK, 40 cols
PriceRange - main_FK, min, max, mm_mean, 3 budgets, 4 budgets (explore a little more)

"""

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

#--- User settings
data_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Merged/" + 
             "LX_RestaurantData_TA_mn10_mx50_Merged_0_4900")

output_path = "/home/jon/GitRepos/LX_Restaurants/Output/Formatted/"

#--- Functions

def split_sort_unique_labels(df_col):
    """
    a. Split multi-label strings with "|" separator, append
    to long list with all labels (incl. duplicates)
    b. Set unique values, replace "_", asecnd sort
    output is all unique labels in alphabetical order
    """
    # Split multi-label strings
    temp_labels = df_col[df_col.notnull()]
    su_labels = []
    temp = [su_labels.append(j.replace(" ", "_")) 
            for i in list(temp_labels) 
            for j in i.split(" | ")]
    
    # set unique values
    su_labels = list(set(su_labels))
    su_labels = [i.replace("_", " ")
                 for i in su_labels]
    su_labels.sort()   
    return su_labels

def binary_label_output(df_col, u_labels):
    """ 
    Outputs a sample * u_label binary matrix
    (1 if label is present for that sample)
    """

    # Get label strings for each sample as a list
    all_str_labels = list(df_col)
    all_str_labels = ["None" if i is None else i
                     for i in all_str_labels]
    
    # Make sample * u_label matrix
    output = np.zeros((len(all_str_labels),len(u_labels)),dtype = "int")
    for u_idx, u in enumerate(u_labels):
        output[:,u_idx] = np.array([1 if u in i else 0 
                                    for i in all_str_labels])  
    return output

def save_binary_mat_df(
                       df_col, ulabels, binary_matrix, 
                       out_head_str, output_path):    
    """ 
    Pickles binary matrix as dataframe along with column keys for
    the corresponding labels
    Exception if indices of old and new df are not identical
    """
    # Create dataframe + column_name key
    col_names = [("%s_%i" % (out_head_str,i)) 
                  for i in np.arange(0,len(ulabels))]
    df_out = pd.DataFrame(
             binary_matrix, 
             columns = col_names).reset_index(names = "Main_PK")
            
    # Pickle data 
    if (len(df_col) == len(df_out) and 
        np.sum(df_col.index - df_out.index) == 0):
    
        pickle_path = os.path.join(output_path,
                                   "%s_Data.pickle" 
                                   % out_head_str)
        col_name_key = ulabels 
        with open(pickle_path,"wb") as f:
            pickle.dump(df_out, f)
            pickle.dump(col_name_key, f)        
    else:
        raise Exception("Data not saved! New df index doesn't match main index.")

def load_binary_mat_df(pickle_path):
    """ Load pickled df and col_key as dict
    """
    lbm_dict = {}
    with open(pickle_path, "rb") as f:
        lbm_dict["df"] = pickle.load(f)
        lbm_dict["col_key"] = pickle.load(f)
    return lbm_dict

#--- MAIN

# Make out_path
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)

# Load scraped data
df_scr = pd.read_pickle(data_path)

#--- Cuisine data


df_col = df_scr.Cuisines
out_head_str = "Cuis"

ulabels = split_sort_unique_labels(df_col)
binary_matrix = binary_label_output(df_col,ulabels)
save_binary_mat_df(
    df_col, ulabels, binary_matrix, 
    out_head_str, output_path
    )

# # Load pickle df and col_key
# pickle_path = os.path.join(output_path,
#                            "%s_Data.pickle" 
#                            % out_head_str)       
# cuis_dict = load_binary_mat_df(pickle_path)
