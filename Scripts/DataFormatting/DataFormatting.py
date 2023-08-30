# 
"""
Takes scraped data df and outputs 8 formatted dataframes:

1. Main (Restaurant info): MainPK, Name, SkipCode, TotalReviewsEN, AvgRating,
                           Food, service, Value, & Atmosphere sub-ratings
2. String (longer text info): MainFK, URL, Address, About, ReviewTags
3. Price (food costs): MainFK, min, max and mid (mean of min-max),
                       3 budget categories based on quantile ranges 
                       (e.g. low, mid, highest third of prices) 
                       for each min, max, mid
                       4 budget categories (as above)

4. Cuisine (binary coding of unique cuisine labels per restaurant, e.g.
            40 unique cuisine labels denoted with 1s or 0s) + Main_FK
5. Diet (binary coding as above)
6. Meal ("")
7. RestFeat (restaurant features; "")   
8. Review (unravelled reviews across resturants): titles, dates, ratings, 
            review text, along with Main_FK, review index (0 - ~69k reviews),
            rev_num (nth review per given restaurant (e.g. 0-49))
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import sys
scripts_path = "/home/jon/GitRepos/LX_Restaurants/Scripts/"
sys.path.append(scripts_path)
from Functions.dataformatting import *

#--- User settings
data_path = ("/home/jon/GitRepos/LX_Restaurants/Output/Merged/" + 
             "LX_RestaurantData_TA_mn10_mx50_Merged_0_4900")

output_path = "/home/jon/GitRepos/LX_Restaurants/Output/Formatted/"

#--- MAIN

# Make out_path
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)

# Load scraped data
df_scr = pd.read_pickle(data_path)

#--- Main data

# Decimalize sub-ratings
df_out = df_scr[["FoodRating", 
                 "ServiceRating", "ValueRating",
                 "AtmosphereRating"]] / 10

# Concatenate Main restaturant data and decimalized sub-ratings
df_out = pd.concat((df_scr[["Name", "SkipCode", "TotalReviewsEN", 
                 "AvgRating"]].reset_index(names = "MainPK"),df_out),
              axis = 1)

# Pickle
pickle_path = os.path.join(output_path,"Main_Data.pickle") 
pickle_df_safe(df_scr, df_out, pickle_path)

#--- Long string data (URL, address, about, ReviewTags)

df_out = df_scr[["URL", "Address", "About", "ReviewTags"]].reset_index(
                                                           names = "MainFK")
# Pickle
pickle_path = os.path.join(output_path,"String_Data.pickle") 
pickle_df_safe(df_scr, df_out, pickle_path)

#--- Price data

# Format min, max, and mid (mean of both) prices
price_range = list(df_scr.PriceRange)
min_max_price = [(np.nan,np.nan) if i is None else 
                 (int(i.split("€")[1][:-3].replace(",", "")), 
                  int(i.split("€")[2].replace(",", "")))
                 for i in price_range]
min_max_price = np.array(min_max_price)
mid_price = np.nanmean(min_max_price, axis = 1)


def quantile_label_no_outliers_idx(val, percentile, qlab):
    """ Create quantile labels to values below percentile and assign 
    values above with inf
    val = mid_price
    percentile = 97.5
    qlab = 3 (new labels)
    
    a. Get indices of numerical values >= given percentile
    b. Nan them (temporarily)
    c. Apply pd.qcut
    d. Assign inf to outlier values
    """
    # Indices of values >= percentile
    val = np.copy(val)
    prcntl = np.nanpercentile(val,percentile)
    o_i = np.where(val >= prcntl)[0]
    
    # Nan    
    val[o_i] = np.nan
    
    # Qcut, set outliers as inf
    qcut = pd.qcut(val,qlab, labels = False)
    qcut[o_i] = np.Inf    
    print("%i values >= %1.1f percentile" % (len(o_i),
        percentile))
    return qcut

# Create qcut labels after handling outliers (values > 97.5 percentile)
mid_cat_2 = quantile_label_no_outliers_idx(mid_price, 97.5, 2)
mid_cat_3 = quantile_label_no_outliers_idx(mid_price, 97.5, 3)
mid_cat_4 = quantile_label_no_outliers_idx(mid_price, 97.5, 4)

min_cat_2 = quantile_label_no_outliers_idx(min_max_price[:,0], 97.5, 2)
min_cat_3 = quantile_label_no_outliers_idx(min_max_price[:,0], 97.5, 3)
min_cat_4 = quantile_label_no_outliers_idx(min_max_price[:,0], 97.5, 4)

max_cat_2 = quantile_label_no_outliers_idx(min_max_price[:,1], 97.5, 2)
max_cat_3 = quantile_label_no_outliers_idx(min_max_price[:,1], 97.5, 3)
max_cat_4 = quantile_label_no_outliers_idx(min_max_price[:,1], 97.5, 4)

""" Assign price columns to df_out
Discretize price values - 3 or 4 categorical labels based on quantile cuts"""

df_out = pd.DataFrame({
                       "MidPrice": mid_price,
                       "MidCat_2": mid_cat_2,
                       "MidCat_3": mid_cat_3,
                       "MidCat_4": mid_cat_4,
                                           
                       "MinPrice": min_max_price[:,0],
                       "MinCat_2": min_cat_2,
                       "MinCat_3": min_cat_3,
                       "MinCat_4": min_cat_4,                     
                       
                       "MaxPrice": min_max_price[:,1],
                       "MaxCat_2": max_cat_2,
                       "MaxCat_3": max_cat_3,
                       "MaxCat_4": max_cat_4,
                       }
                      ).reset_index(names = "MainFK")

# Pickle
pickle_path = os.path.join(output_path,"Price_Data.pickle") 
pickle_df_safe(df_scr, df_out, pickle_path)

#--- Cuisine data

df_col = df_scr.Cuisines
out_head_str = "Cuisine"

ulabels = split_sort_unique_labels(df_col)
binary_matrix = binary_label_output(df_col,ulabels)
save_bin_pickled_df_safe(
    df_col, ulabels, binary_matrix, 
    out_head_str, output_path
    )

#--- Special diets

df_col = df_scr.SpecialDiets
out_head_str = "Diet"

ulabels = split_sort_unique_labels(df_col)
binary_matrix = binary_label_output(df_col,ulabels)
save_bin_pickled_df_safe(
    df_col, ulabels, binary_matrix, 
    out_head_str, output_path
    )

#--- Meals

df_col = df_scr.Meals
out_head_str = "Meal"

ulabels = split_sort_unique_labels(df_col)
binary_matrix = binary_label_output(df_col,ulabels)
save_bin_pickled_df_safe(
    df_col, ulabels, binary_matrix, 
    out_head_str, output_path
    )

#--- Features

df_col = df_scr.Features
out_head_str = "RestFeat"

ulabels = split_sort_unique_labels(df_col)
binary_matrix = binary_label_output(df_col,ulabels)
save_bin_pickled_df_safe(
    df_col, ulabels, binary_matrix, 
    out_head_str, output_path
    )

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

# Create HiLowRatings (2 if > 3, 1 if < 3, 0 else)
hilowratings = np.array([2 if i > 3 else 1 if i < 3 else 0
                         for i in list(ratings)])

# Format dates
dates = np.hstack(np.asarray(dates))
dates = [datetime.strptime(dates[i], '%B %d, %Y') if dates[i] != " " else datetime.strptime("January 1, 1970", '%B %d, %Y')
     for i in np.arange(len(dates))]

# Pickle
df_out = pd.DataFrame({"MainFK" : main_FK, 
                       "RevNum": rev_num,
                       "RevTitle": titles,
                       "RevDate": dates,
                       "RevRating": ratings,
                       "RevHiLow": hilowratings,
                       "RevText": texts}).reset_index(names = "RevIdx")

# Remove empty values (wiht no rating)
df_out = df_out[df_out.RevRating > 0]

pickle_path = os.path.join(output_path,"Review_Data.pickle")
with open(pickle_path,"wb") as f:
    pickle.dump(df_out, f)

#--- Load checks

# Binary data
print("Data checks:")
data_names_bin = ["Cuisine", "Diet", "Meal", "RestFeat"]
for d in data_names_bin:    
    check_dict = load_bin_pickled_df(
                 os.path.join(output_path,"%s_Data.pickle" % d))
    print("df %s shape: %i %i" % (d, check_dict["df"].shape[0],
          check_dict["df"].shape[1]))
    print("%s labels: %i + FK" % (d, len(check_dict["col_key"])))
del check_dict

# Other data
data_names = ["Main", "String", "Price", "Review"]
for d in data_names:    
    check_df = load_pickled_df(
                 os.path.join(output_path,"%s_Data.pickle" % d))
    print("df %s shape: %i %i" % (d, check_df.shape[0],
          check_df.shape[1]))
del check_df


