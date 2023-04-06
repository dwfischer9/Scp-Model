import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer

stop_words = ['the', 'and', 'to', 'of', 'is', 'in', 'that', 'it', 'with', 'for', 'on', 'was', 'as', 'at', 'by', 'an', 'be', 'this', 'which', 'or', 'from', 'not', 'but',
              'are', 'they', 'if', 'we', 'all', 'can', 'more', 'will', 'has', 'their', 'its', 'who', 'than', 'then', 'had', 'her', 'him', 'she', 'he', 'my', 'your', 'our', 'us']


def readData():
    df = pd.read_csv('../data/scp6999.csv')
    print("Successfully read data from CSV.")
    print("Now cleaning the data...")
    # cleaning up data, all text including and after this is not part of the article.
    df['text'] = df['text'].str.split('« SCP').str[0]
    df['text'] = df['text'].replace('[^a-zA-Z\s]', '', regex=True)
    df['text'] = df['text'].replace('  ', ' ', regex=True)
    df['text'] = df['text'].str.lower().apply(lambda x: ' '.join(
        [word for word in x.split() if word not in stop_words]))  # remove stop words and put to lowercase
    df['Euclid'] = df['Euclid'].fillna('none/other')
    stemmer = PorterStemmer()
    df['text_stemmed'] = df['text'].apply(
        lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))  # convert words to their stemmed form
    return df
