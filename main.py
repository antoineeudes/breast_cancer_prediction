import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Get X, Y arrays from dataframe
    df = pd.read_csv('data.csv')
    Y = np.array(df['diagnosis'] == 'M').astype(int)
    X = np.array(df)[:, 2:-1]