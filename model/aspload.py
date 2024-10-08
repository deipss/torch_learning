

import  pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../data/ASP_2010-01_0100.txt', sep='\t', encoding='utf-8')
    print(df.head())



