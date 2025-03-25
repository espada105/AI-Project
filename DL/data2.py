import pandas as pd

data = {
    'Name': ['Tom', 'Brian', 'Alice'],
    'Age': [10, 15, 20],
    'Score': [90, 85, 95]
}

df = pd.DataFrame(data, index=['a', 'b', 'c'])

print(df.loc['a'])  
print(df.loc['a', 'Name'])  
print(df.loc[['a', 'b'], ['Name', 'Score']])  

print(df[df['Name'].str.lower().str.startswith('a')])

result = df.loc[df['Age']>=30,'Name']
print(result)


# b bob
# c charilie
# name: Name, dtype: object
import pandas as pd

data = {
    'Name': ['Tom', 'bob', 'charilie'],
    'Age': [10, 35, 40],
    'Score': [90, 85, 95]
}

df = pd.DataFrame(data, index=['a', 'b', 'c'])


result = df.loc[df['Age']>=30, 'Name']
print(result)