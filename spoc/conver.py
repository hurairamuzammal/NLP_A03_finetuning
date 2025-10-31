import pandas as pd

import openpyxl
df=pd.read_csv('train\\spoc-train-py.tsv', sep="\t",)

df.to_excel('spoc-train-py.xlsx', index=False)


print("Conversion Completed")