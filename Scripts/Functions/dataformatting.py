import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
    _ = [su_labels.append(j.replace(" ", "_")) 
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

def pickle_df_safe(df_orig, df_out, pickle_path):
    """ Pickle the output df but ensure that is has the same
    indices as the original df
    """
    if (len(df_orig) == len(df_out) and 
        np.sum(df_orig.index - df_out.index) == 0):
    
        with open(pickle_path,"wb") as f:
            pickle.dump(df_out, f)
    else:
        raise Exception("Data not saved! New df index doesn't match main index.")


def save_bin_pickled_df_safe(
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

def load_bin_pickled_df(pickle_path):
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

def n_perc_vals(data,n_bounds):
    """ Returns values for each linearly spaced percentile
    interval (e.g. 5 = 0,25,50,75,100)
    """
    b = np.linspace(0,100,n_bounds)
    b_vals = [np.percentile(data[~np.isnan(data)], i)
              for i in b]
    return b_vals

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

def strat_split_by_rating_75(df_rev_path):
    """
    Stratified split review data
    Hardcoded for 75% training 25% testing split
    """
    # Load review data
    df_rev = load_pickled_df(df_rev_path)
    
    # KStratSamples
    X = df_rev.RevTitle
    y = df_rev.RevRating
    skf = StratifiedKFold(n_splits=4, random_state = 42, shuffle = True)
    skf.get_n_splits(X, y)
    
    # Get first fold, get train and test indices
    fold_A,_,_,_ = skf.split(X,y)
    tr_i = fold_A[0]
    te_i = fold_A[1]
    
    # # Print the proportion of tr/te samples per rating
    # for r in np.arange(5,0,-1):
    #     total_per_rating = len(df_rev[df_rev.RevRating == r])
    #     tr_prop = (len(df_rev[(df_rev.RevRating == r) & 
    #                (df_rev.RevIdx.isin(tr_i))]) / 
    #                len(df_rev[df_rev.RevRating == r]))
    #     te_prop = (len(df_rev[(df_rev.RevRating == r) & 
    #                (df_rev.RevIdx.isin(te_i))]) / 
    #                len(df_rev[df_rev.RevRating == r]))
    #     print("%i star train prop = %1.2f of total %s samples" % 
    #           (r,tr_prop, total_per_rating))
    #     print("%i star test prop = %1.2f of total %s samples" % 
    #           (r,te_prop, total_per_rating))
    return tr_i, te_i

def strat_split_by_rating_50(df_rev_path):
    """
    Stratified split review data
    Hardcoded for 7%0 training 50% testing split
    """
    # Load review data
    df_rev = load_pickled_df(df_rev_path)
    
    # KStratSamples
    X = df_rev.RevTitle
    y = df_rev.RevRating
    skf = StratifiedKFold(n_splits=2, random_state = 42, shuffle = True)
    skf.get_n_splits(X, y)
    
    # Get first fold, get train and test indices
    fold_A,_ = skf.split(X,y)
    tr_i = fold_A[0]
    te_i = fold_A[1]
    
    # # Print the proportion of tr/te samples per rating
    # for r in np.arange(5,0,-1):
    #     total_per_rating = len(df_rev[df_rev.RevRating == r])
    #     tr_prop = (len(df_rev[(df_rev.RevRating == r) & 
    #                (df_rev.RevIdx.isin(tr_i))]) / 
    #                len(df_rev[df_rev.RevRating == r]))
    #     te_prop = (len(df_rev[(df_rev.RevRating == r) & 
    #                (df_rev.RevIdx.isin(te_i))]) / 
    #                len(df_rev[df_rev.RevRating == r]))
    #     print("%i star train prop = %1.2f of total %s samples" % 
    #           (r,tr_prop, total_per_rating))
    #     print("%i star test prop = %1.2f of total %s samples" % 
    #           (r,te_prop, total_per_rating))
    return tr_i, te_i