from sklearn.model_selection import train_test_split
from process_data import *
from build_features import *
from train_model import *
from evaluate_model import *
import pandas as pd
import numpy as np

def main():
    print("Loading data from CSV...")

    df_raw = readData()
    print(df_raw.head())
    print(df_raw.shape)
    print(df_raw.columns)
    print(df_raw.describe())
    print("Data has been loaded and cleaned!")
    print("Building features...")
    df = df_raw
    X, unique_words = bag_of_words(df['text'])
    
    print("Word frequency features have been built.")
    # df.to_csv("../data/processed/freq.csv", index=False)
    # print("Data has been saved to data/processed/freq.csv")
    # df = pd.read_csv("../data/processed/freq.csv")
    y = df['Euclid']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    trainedModel = train_model(X_train, y_train)
    test_Model(trainedModel, X_test, y_test)

if __name__ == "__main__":
    main()
