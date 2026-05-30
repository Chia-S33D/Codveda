# 17 May 2026
# Task 1: Data cleaning and preprocessing
# Needs just python and pandas

import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

df = pd.read_csv(r"C:\Users\User\OneDrive\UCT\Year 3\Codveda\Data Set For Task-2026\Data Set For Task\2) Stock Prices Data Set.csv")
print(df.head())
print(df.info())
print(df.dtypes)

print(df.duplicated().sum()) # none
 
# dftypes shows us that date is a strin which is an incorrect format that needs to be changed to date-time format using pandas
df['date'] = pd.to_datetime(df['date'])
print(df.dtypes) # test

# df.info() shows us that columns 'open', 'high', and 'low' have null values.
# We need to delete all rows with null values
df_clean = df.dropna()
print(df_clean.info())


### What was done: 
# - corrected incorrect formatting
# - deleted rows with missing values
# - checked for duplicates and found none.