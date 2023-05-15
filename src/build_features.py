from collections import Counter
import pandas as pd
import numpy as np
from process_data import readData
import warnings
warnings.filterwarnings("ignore")


def bag_of_words(texts):
    """
    Computes a bag of words representation for a list of texts
    """
    # Preprocess the texts
    preprocessed_texts = texts

    # Create a dictionary of word frequencies
    word_freqs = Counter()
    for text in preprocessed_texts:
        words = text.split()
        word_freqs.update(words)

    # Create a list of unique words
    unique_words = list(word_freqs.keys())

    # Create a dictionary mapping words to indices
    word_to_index = {word: i for i, word in enumerate(unique_words)}

    # Create a matrix of word frequencies
    num_texts = len(preprocessed_texts)
    num_words = len(unique_words)
    word_freq_matrix = [[0 for j in range(num_words)] for i in range(num_texts)]
    for i, text in enumerate(preprocessed_texts):
        words = text.split()
        for word in words:
            word_index = word_to_index[word]
            word_freq_matrix[i][word_index] += 1

    return word_freq_matrix, unique_words