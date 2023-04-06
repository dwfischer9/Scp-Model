import pandas as pd
import numpy as np
from process_data import readData
import warnings
warnings.filterwarnings("ignore")


def calcWordFreq(df):
    df = pd.concat([df, df['text_stemmed'].apply(
        get_word_freq).apply(pd.Series)], axis=1)
    print(df.head().fillna(0))
    df = df.drop(['text_stemmed'], axis=1)
    return df.fillna(0)


def get_word_freq(row):
    words = row.split()
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    total_words = len(words)
    rel_freq = {k: v / total_words for k, v in freq.items()}
    return rel_freq
