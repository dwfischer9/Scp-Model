from process_data import *
from build_features import *


def main():
    print("Loading data from CSV...")
    df_raw = readData()
    print("Data has been loaded and cleaned!")
    df_freq = calcWordFreq(df_raw)
    
if __name__ == "__main__":
    main()