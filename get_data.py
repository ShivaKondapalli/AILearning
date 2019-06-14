import pandas as pd

path = 'https://raw.githubusercontent.com/WillKoehrsen/Bokeh-Python-Visualization/master/bokeh_app/data/flights.csv'
df = pd.read_csv(path, sep=',')

print(df.head(15))
print(df.shape)

df = df.drop('Unnamed: 0', axis=1)

df.to_csv('data/flights.csv')


