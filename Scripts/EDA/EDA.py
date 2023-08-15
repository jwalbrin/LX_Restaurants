# EDA for LX_restaurant data

import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import sys
scripts_path = "/home/jon/GitRepos/LX_Restaurants/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import load_bin_pickled_df, load_pickled_df

#--- User settings
data_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Formatted/")
     
#--- MAIN

#--- Functions
# Get indices for filter lab
def get_filt(filt_dict, filt_lab):
    """ For subsequent filtering, get: 
    a. Sparse binary column for a given filter label from a 
    dataset (e.g. "portuguese" column from Cuisine_Data.pickle)
    b. The row indices of all 1s from a
    """    
    # Get filt data name, find column index, get values
    filt_name = filt_dict["df"].columns[1].split("_")[0]
    targ_id = filt_dict["col_key"].index(filt_lab)
    filt_bin = filt_dict["df"][filt_name + "_%i" % targ_id]
    filt_idx = np.where(filt_bin==1)[0]
    print("Label and column name: %s, %s" % (filt_dict["col_key"][targ_id],
          filt_bin.name))
    return filt_bin, filt_idx

# Load data
df = load_pickled_df(data_path + "Main_Data.pickle")

#--- Filter portuguese (binary)
filt_lab = "portuguese"
filt_name = "Cuisine"

# Load df and col keys
filt_pickle = (data_path + ("%s_Data.pickle" % filt_name))
filt =load_bin_pickled_df(filt_pickle)

# Get filter variables
pt_filt_b, pt_filt_i = get_filt(filt, filt_lab)
del filt

#--- Filter Price (non-nan)
df_pr = load_pickled_df(data_path + "Price_Data.pickle")
pr_filt_i = np.where(~np.isnan(df_pr.MidCat_3)==1)[0]

# Load rev data, filter date, drop cols
df_rev = load_pickled_df(data_path + "Review_Data.pickle")
df_rev = df_rev[df_rev.RevDate > "2021-01-01 00:00:00"]
df_rev.drop(["RevTitle", "RevRating"],axis = 1,inplace = True)

df_rev = df_rev[(df_rev["MainFK"].isin(pt_filt_i)) & 
                (df_rev["MainFK"].isin(pr_filt_i))]




# Join price data
df_pr = load_pickled_df(data_path + "Price_Data.pickle")

a = df_rev[["MainFK", "RevNum"]]
b = df_pr[["MainFK", "MidCat_3"]]
c = a.join(b, on = "MainFK",lsuffix='_caller', rsuffix='_other',
           validate = "m:1")




# #--- Frequency of unique labels
# # Meals
# meals = split_unique_labels(df.Meals)
# counts, words = desc_freq_words(meals)
# top_n = np.arange(0,len(words))
# label_type = "Meal"
# countplot_unique_labels(words, counts,top_n,label_type)

# # Special diets
# diets = split_unique_labels(df.SpecialDiets)
# counts, words = desc_freq_words(diets)
# top_n = np.arange(0,len(words))
# label_type = "Special Diet"
# countplot_unique_labels(words, counts,top_n,label_type)

# # Restaurant features
# rest_features = split_unique_labels(df.Features)
# counts, words = desc_freq_words(rest_features)
# top_n = np.arange(0,len(words))
# label_type = "Rest. Feats."
# countplot_unique_labels(words, counts,top_n,label_type)

# # Cuisines
# cuisines = split_unique_labels(df.Cuisines)
# counts, words = desc_freq_words(cuisines)
# top_n = np.arange(0,30)
# label_type = "Cuisine"
# countplot_unique_labels(words, counts,top_n,label_type)

# #--- Price range 
# # Format min, max, and mid (mean of both) prices
# price_range = list(df.PriceRange)
# min_max_price = [(np.nan,np.nan) if i is None else 
#                  (int(i.split("€")[1][:-3].replace(",", "")), 
#                   int(i.split("€")[2].replace(",", "")))
#                  for i in price_range]
# min_max_price = np.array(min_max_price)
# mid_price = np.nanmean(min_max_price, axis = 1)

# # Plot        
# hist_with_percentiles(mid_price, "Mid Price", [33,66])
# hist_with_percentiles(min_max_price[:,0], "Min Price", [33,66])
# hist_with_percentiles(min_max_price[:,1], "Max Price", [33,66])

# #--- Ratings: Frequencies
# rating_names = ["AvgRating","FoodRating", "ServiceRating",
#                 "ValueRating", "AtmosphereRating"]
# sp_idx = [(0,0),(0,1),(0,2),
#           (1,0),(1,1)]
# cm = 1/2.54
# fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14*cm, 10*cm), dpi = 100,
#                        sharey=True)
# plt.subplots_adjust(hspace=1)
# plt.suptitle("Rating Frequencies", x = 0.5, y = 1.1)
# for sp, r in enumerate(rating_names):
#     sns.histplot(df, x = r, ax = ax[sp_idx[sp]])       

# print("Valid Average Ratings: %i" % np.sum(df.AvgRating.notnull()))
# print("Valid Food Ratings: %i" % np.sum(df.FoodRating.notnull()))
# print("Valid Service Ratings: %i" % np.sum(df.ServiceRating.notnull()))
# print("Valid Value Ratings: %i" % np.sum(df.ValueRating.notnull()))
# print("Valid Atmosphere Ratings: %i" % np.sum(df.AtmosphereRating.notnull()))


# # # pair grod
# # # Map to upper,lower, and diagonal
# # g = sns.PairGrid(df)
# # g.map_diag(plt.hist)
# # g.map_upper(plt.scatter)
# # g.map_lower(sns.kdeplot)





