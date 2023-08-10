# EDA for LX_restaurant data

import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


#--- User settings
data_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Merged/" + 
             "LX_RestaurantData_TA_mn10_mx50_Merged_0_4900")

#--- Functions

def split_unique_labels(df_col):
    """
    Split multi-label strings with "|" separator, append
    to long list with all labels (incl. duplicates)
    """
    
    all_labels = df_col[df_col.notnull()]
    su_labels = []
    temp = [su_labels.append(j.replace(" ", "_")) 
            for i in list(all_labels) 
            for j in i.split(" | ")]
    return su_labels

def desc_freq_words(in_list):
    """
    Use count vecotrizer to plot words by descending frequency
    """

    this_list = in_list
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(this_list)
    total_words = vectorizer.get_feature_names_out()
    total_counts = np.sum(X.toarray(), axis=0)
    desc_idx = np.argsort((-total_counts)) # negated array for descend sort
    desc_counts = total_counts[desc_idx] 
    desc_words = total_words[desc_idx] 
    return desc_counts, desc_words

def countplot_unique_labels(words, counts,top_n,label_type):
    plt.figure(figsize=(15,5))
    bars = sns.barplot(
        x=words[top_n],
        y=counts[top_n])
    bars.set_xticklabels(bars.get_xticklabels(), rotation=90)
    bars.set_ylabel("Count")
    bars.set_title("%i most frequent %s labels (of %i)" 
                   % (len(top_n), label_type, len(words)))
    
def hist_with_percentiles(data, label, percentiles):
    fig, ax = plt.subplots()
    hist = sns.histplot(data, binrange = (0,200))
    hist.set_title(label)
    for p in percentiles:
        prcntl = np.percentile(data[~np.isnan(data)], p) 
        plt.axvline(prcntl, 0,1, c = 'r', alpha = 0.5)
#--- MAIN

# Load data
df = pd.read_pickle(data_path)

#--- Frequency of unique labels
# Meals
meals = split_unique_labels(df.Meals)
counts, words = desc_freq_words(meals)
top_n = np.arange(0,len(words))
label_type = "Meal"
countplot_unique_labels(words, counts,top_n,label_type)

# Special diets
diets = split_unique_labels(df.SpecialDiets)
counts, words = desc_freq_words(diets)
top_n = np.arange(0,len(words))
label_type = "Special Diet"
countplot_unique_labels(words, counts,top_n,label_type)

# Restaurant features
rest_features = split_unique_labels(df.Features)
counts, words = desc_freq_words(rest_features)
top_n = np.arange(0,len(words))
label_type = "Rest. Feats."
countplot_unique_labels(words, counts,top_n,label_type)

# Cuisines
cuisines = split_unique_labels(df.Cuisines)
counts, words = desc_freq_words(cuisines)
top_n = np.arange(0,30)
label_type = "Cuisine"
countplot_unique_labels(words, counts,top_n,label_type)

#--- Price range 
# Format min, max, and mid (mean of both) prices
price_range = list(df.PriceRange)
min_max_price = [(np.nan,np.nan) if i is None else 
                 (int(i.split("€")[1][:-3].replace(",", "")), 
                  int(i.split("€")[2].replace(",", "")))
                 for i in price_range]
min_max_price = np.array(min_max_price)
mid_price = np.nanmean(min_max_price, axis = 1)

# Plot        
hist_with_percentiles(mid_price, "Mid Price", [33,66])
hist_with_percentiles(min_max_price[:,0], "Min Price", [33,66])
hist_with_percentiles(min_max_price[:,1], "Max Price", [33,66])

#--- Ratings: Frequencies
rating_names = ["AvgRating","FoodRating", "ServiceRating",
                "ValueRating", "AtmosphereRating"]
sp_idx = [(0,0),(0,1),(0,2),
          (1,0),(1,1)]
cm = 1/2.54
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14*cm, 10*cm), dpi = 100,
                       sharey=True)
plt.subplots_adjust(hspace=1)
plt.suptitle("Rating Frequencies", x = 0.5, y = 1.1)
for sp, r in enumerate(rating_names):
    sns.histplot(df, x = r, ax = ax[sp_idx[sp]])       

print("Valid Average Ratings: %i" % np.sum(df.AvgRating.notnull()))
print("Valid Food Ratings: %i" % np.sum(df.FoodRating.notnull()))
print("Valid Service Ratings: %i" % np.sum(df.ServiceRating.notnull()))
print("Valid Value Ratings: %i" % np.sum(df.ValueRating.notnull()))
print("Valid Atmosphere Ratings: %i" % np.sum(df.AtmosphereRating.notnull()))


# # pair grod
# # Map to upper,lower, and diagonal
# g = sns.PairGrid(df)
# g.map_diag(plt.hist)
# g.map_upper(plt.scatter)
# g.map_lower(sns.kdeplot)





