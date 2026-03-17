import pandas as pd

train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")

print(pd.head())
print(pd.shape)
print(pd.columns)
print(pd.isnull().sum())