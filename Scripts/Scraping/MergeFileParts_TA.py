# Merge and format scraped part files
# Note: Point to directory that ONLY contains desired files to be merged!!!

import numpy as np
import pandas as pd
import os
import pickle

input_path = "/home/jon/GitRepos/LX_Restaurants/Output/FileParts/"
output_path = "/home/jon/GitRepos/LX_Restaurants/Output/Merged/"
file_prefix = "LX_RestaurantData_TA_mn10_mx50_"

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

#--- Suffix duplicate names with numbers
# Find duplicate names
df_dups = (df.groupby("Name", as_index = False, sort = False)
               ["URL"].count())
df_dups = df_dups[df_dups.URL > 1]

# Suffix duplicates with a number based on their ordering in df 
# (e.g. 3rd suplicate will have [3])
dup_names = list(df_dups.Name)
dup_idx = [np.array(df.query("Name == @i").index) 
                for i in dup_names]
all_names = list(df.Name)
for dn_i, dn in enumerate(dup_names):    
    for di_i, di in enumerate(dup_idx[dn_i]):
        all_names[di] = all_names[di] + " [" + str(di_i) + "]"        
df.Name = all_names

#--- Find missing names
df_missing = df[df.Name.isnull()]

# Pickle & csv
output_name = (file_prefix + "Merged_" + str(np.min(pf_start_idx)) + "_" +
            str(np.max(pf_start_idx)))
pickle_file = open(os.path.join(output_path, output_name), "wb")
pickle.dump(df,pickle_file)
pickle_file.close()
df.to_csv(os.path.join(output_path, output_name + ".csv"))

#--- Checks

# load pickled file, check length, N unique URLs
df_check = pd.read_pickle(os.path.join(output_path, output_name))
len(df_check)
df_check.URL.nunique()

# Check N reviews is correct 
# TotalReviewsEN and n reviews, respectively
df_rev_check = df_check.query("TotalReviewsEN > 0").reset_index()
for i in np.arange(100,200):
    try:
       print(df_rev_check.TotalReviewsEN.at[i],   
             len(set(df_rev_check.ReviewTexts.at[i])))
    except: print("Skip Index: " + str(i))

# Counts of restaurants with more than 50 En reviews
print(len(df_rev_check.query("TotalReviewsEN >= 50")),
      len(df_rev_check.query("TotalReviewsEN >= 40")),
      len(df_rev_check.query("TotalReviewsEN >= 30")),
      len(df_rev_check.query("TotalReviewsEN >= 20")),
      len(df_rev_check.query("TotalReviewsEN >= 10")))

# Unique checks
df_check.URL.nunique()
df_check.Name.nunique()
df_check.Name.unique()




        
        
        
        