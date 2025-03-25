import pandas as pd

data = {'Name': ['Alice','Bob',None], 
        'Age': [25, None, 30]}

df = pd.DataFrame(data)

# df['Age'] = df['Age'].fillna(0)
# print(df)

# # df['Name'] = df['Name'].fillna('Unknown')
# # print(df)

# df.loc[df['Name'].isna(), 'Name'] = 'Unknown'
# print(df)


cleaned_df = df.dropna()
print(cleaned_df)

cleaned_df = df.dropna(axis=1)
print(cleaned_df)