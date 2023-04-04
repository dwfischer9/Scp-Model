import pandas as pd
import numpy as np
from process_data import readData


def calcWordFreq(df):
    df = pd.concat([df, df['text'].apply(
        get_word_freq).apply(pd.Series)], axis=1)
    print(df.head())
    return df


def get_word_freq(row):
    words = row.split()
    freq = {}
    for word in words:
        print(word)
        freq[word] = freq.get(word, 0) + 1
    total_words = len(words)
    rel_freq = {k: v / total_words for k, v in freq.items()}
    return rel_freq
