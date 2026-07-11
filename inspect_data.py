import pandas as pd

df = pd.read_csv('Car_Insurance_Claim.csv')
print(df.head().to_string())
print(df.columns.tolist())
print(df.shape)
print(df.dtypes)
