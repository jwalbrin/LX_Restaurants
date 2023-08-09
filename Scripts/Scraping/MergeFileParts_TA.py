# Merge and format scraped part files
# Note: Point to directory that ONLY contains desired files to be merged

import numpy as np
import pandas as pd
import os
import pickle

#--- User settings
input_path = "/home/jon/GitRepos/LX_Restaurants/Output/FileParts/"
output_path = "/home/jon/GitRepos/LX_Restaurants/Output/Merged/"
file_prefix = "LX_RestaurantData_TA_mn10_mx50_"

#--- Functions

def suffix_duplicates(df):
    """  
    a. Finds duplicates ( i.e. multi branch restaurant entries)
    b. Finds indices for all duplicates per restaurant and 
       suffixes those entries in df (e.g. "[1]" etc.) 
    """
    
    # Find duplicate names
    df_dups = (df.groupby("Name", as_index = False, sort = False)
                   ["URL"].count())
    df_dups = df_dups[df_dups.URL > 1]
    print("There are %i restaurants with multiple branches" % len(df_dups))

    # Suffix duplicates (e.g. 3rd duplicate will have [3])
    dup_names = list(df_dups.Name)
    dup_idx = [np.array(df.query("Name == @i").index) 
                    for i in dup_names]
    
    all_names = list(df.Name)
    for dn_i, dn in enumerate(dup_names):    
        for di_i, di in enumerate(dup_idx[dn_i]):
            all_names[di] = all_names[di] + " [" + str(di_i) + "]"        
    df.Name = all_names
    
    return df

#--- MAIN

# Make output directory
if os.path.isdir(output_path) == False:
    os.mkdir(output_path)

# Ascending sort part file names
pf_names = os.listdir(input_path)
pf_start_idx = np.array([int(i.split(file_prefix)[-1].split("_")[0]) 
                         for i in pf_names])
pf_sort_idx = np.argsort(pf_start_idx)
pf_names = np.array(pf_names)[pf_sort_idx].tolist()

# Concatenate part files as single df
df = pd.read_pickle(os.path.join(input_path, pf_names[0]))
for pf in pf_names[1:]:
    df_temp = pd.read_pickle(os.path.join(input_path, pf))
    df = pd.concat([df,df_temp])
df.reset_index(inplace = True, drop = True)

# Suffix duplicate restaurant names
df = suffix_duplicates(df)

#--- Find missing names, update df
df_miss = df[df.Name.isnull()]
if len(df_miss) > 0:
    miss_idx = df_miss.index
    miss_names = [(df_miss.iloc[i].URL.split("-Reviews-")[1].split("-Lisbon_Lisbon")[0].replace("_"," "))
                  for i in np.arange(len(miss_idx))]
    df.loc[df.Name.isnull(), "Name"] = miss_names

# Pickle & csv
output_name = (file_prefix + "Merged_" + str(np.min(pf_start_idx)) + "_" +
            str(np.max(pf_start_idx)))
pickle_file = open(os.path.join(output_path, output_name), "wb")
pickle.dump(df,pickle_file)
pickle_file.close()
df.to_csv(os.path.join(output_path, output_name + ".csv"))

#--- Perform some sanity checks

# load pickled file
df_check = pd.read_pickle(os.path.join(output_path, output_name))

# For a subset of restaurants, print total number of EN reviews and available
# number of selected reviews (else print the index for entries that don't
# meet these criteria)
df_rev_check = df_check.query("TotalReviewsEN > 0").reset_index()
print("Showing total EN reviews and N scraped review pairs:")
for i in np.arange(100,200):
    try:
       print(df_rev_check.TotalReviewsEN.at[i],   
             len(set(df_rev_check.ReviewTexts.at[i])))
    except: print("Skip Index: " + str(i))

# Counts of restaurants with at least 50, 40, 30, 20, 10 EN reviews
print("N restaurants with at least: 50, 40, 30, 20, 10 reviews:")
print(len(df_rev_check.query("TotalReviewsEN >= 50")),
      len(df_rev_check.query("TotalReviewsEN >= 40")),
      len(df_rev_check.query("TotalReviewsEN >= 30")),
      len(df_rev_check.query("TotalReviewsEN >= 20")),
      len(df_rev_check.query("TotalReviewsEN >= 10")))

# Unique checks
print("Length of df: %i" % len(df_check))
print("N unique URLs: %i" % df_check.URL.nunique())
print("N unique Names: %i" % df_check.Name.nunique())





        
        
        
        