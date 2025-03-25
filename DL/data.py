import pandas as pd
data = {
    'Name' : ['alice','bob'],
    'Age': [25, 24]
}

df = pd.DataFrame(data)
# print(df)

# print(df.head())
# print(df.tail())
# print(df.info())
# print(df.describe())

# print(df['Name'])
# print(df[['Name','Age']])


filtered_df = df[df['Age'] >= 25]
print(filtered_df)