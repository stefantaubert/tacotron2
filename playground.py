import pandas as pd
import numpy as np
s = "/datasets/models/taco2pt_v2/ds/thchs_v5/A23/filelist.csv"
data = pd.read_csv(s, header=None, sep="\t")
x = float(data.iloc[:, [3]].sum(axis=0)) / 60
print(x)

wavpath_col = 1
symbols_str_col = 2
data = data.iloc[:,[wavpath_col, symbols_str_col]]
print(data.values)