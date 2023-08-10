import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

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