# 
"""

main - PK, Name, SkipCode, TotalReviews, 5 ratings, cuisine_FK (to cuisines PK),
       special

review - main_FK, title, date, rating, text
cuisine - main_FK, 110 cols
Special diets - main_FK, 4 cols
Meals - main_FK, 6 cols
Features - main_FK, 40 cols

long_string_info - main_FK, URL, street_address, about
tags_review - 
PriceRange - main_FK, min, max, mm_mean, 3 budgets, 4 budgets (explore a little more)

"""

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime

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
             columns = col_names).reset_index(names = "Main_FK")
            
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

def load_pickled_df(pickle_path):
    # Simple load single pickled df
    with open(pickle_path,"rb") as f:
         df = pickle.load(f)
    return df

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
#                             "%s_Data.pickle" 
#                             % out_head_str)       
# cuis_dict = load_binary_mat_df(pickle_path)

#--- Special diets

df_col = df_scr.SpecialDiets
out_head_str = "Diet"

ulabels = split_sort_unique_labels(df_col)
binary_matrix = binary_label_output(df_col,ulabels)
save_binary_mat_df(
    df_col, ulabels, binary_matrix, 
    out_head_str, output_path
    )

#--- Meals

df_col = df_scr.Meals
out_head_str = "Meal"

ulabels = split_sort_unique_labels(df_col)
binary_matrix = binary_label_output(df_col,ulabels)
save_binary_mat_df(
    df_col, ulabels, binary_matrix, 
    out_head_str, output_path
    )

#--- Features

df_col = df_scr.Features
out_head_str = "RestFeat"

ulabels = split_sort_unique_labels(df_col)
binary_matrix = binary_label_output(df_col,ulabels)
save_binary_mat_df(
    df_col, ulabels, binary_matrix, 
    out_head_str, output_path
    )

# Load pickle df and col_key
pickle_path = os.path.join(output_path,
                            "%s_Data.pickle" 
                            % out_head_str)       
test_dict = load_binary_mat_df(pickle_path)
df_test = test_dict["df"]
ulab_test = test_dict["col_key"]

#--- Review data

""" Unravel lists so that there are that there will now be
M * N indices/ rows where:
M is restaurants with valid reviews
N is valid reviews per restaurant
In short, around 69k rows
DOes this for review titles, dates, ratings, texts
Also for: 
a) main_FK(row index for df_scr / df_main, 0-4900ish)
b) rev_idx (review index per restaurant, 0-49ish)
"""

# Subset rows with valid reviews, review columns only
df_rev = df_scr[df_scr.SkipCode==0]
df_rev = df_rev[["ReviewTitles", "ReviewDates", 
                 "ReviewRatings", "ReviewTexts"]]

# unravel data for each row in df_rev
main_FK, rev_num = [], []
titles, dates, ratings, texts = [],[],[],[]

_ = [(
      main_FK.append(np.repeat(i,len(df_rev.loc[i].ReviewRatings))),
      rev_num.append(np.arange(len(df_rev.loc[i].ReviewRatings))),
      titles.append(df_rev.loc[i].ReviewTitles),
      dates.append(df_rev.loc[i].ReviewDates),
      ratings.append(df_rev.loc[i].ReviewRatings),
      texts.append(df_rev.loc[i].ReviewTexts)     
     )
     for i in df_rev.index]

# Flatten into 1D arrays
main_FK = np.hstack(np.asarray(main_FK))
rev_num = np.hstack(np.asarray(rev_num))
titles = np.hstack(np.asarray(titles))
texts = np.hstack(np.asarray(texts))

# Format ratings
ratings = np.hstack(np.asarray(ratings)) / 10
ratings = ratings.astype("int")

# Format dates
dates = np.hstack(np.asarray(dates))
dates = [datetime.strptime(dates[i], '%B %d, %Y') if dates[i] != " " else datetime.strptime("January 1, 1970", '%B %d, %Y')
     for i in np.arange(len(dates))]



# Pickle
df_out = pd.DataFrame({"main_FK" : main_FK, 
                       "rev_num": rev_num,
                       "rev_title": titles,
                       "rev_date": dates,
                       "rev_rating": ratings,
                       "rev_text": texts}).reset_index(names = "rev_idx")

pickle_path = os.path.join(output_path,"Rev_Data.pickle")
with open(pickle_path,"wb") as f:
    pickle.dump(df_out, f)

# df_col = df_scr.ReviewDates[df_scr.SkipCode==0]
# main_FK = []
# rev_idx = []
# _ = [(main_FK.append(np.repeat(i,len(df_col[i]))),
#      rev_idx.append(np.arange(len(df_col[i]))))
#      for i in df_col.index]
# main_FK = np.hstack(np.asarray(main_FK))
# rev_idx = np.hstack(np.asarray(rev_idx))








# get non skip row indices, append for all repeats, tuple with the counts for repeats
# 

