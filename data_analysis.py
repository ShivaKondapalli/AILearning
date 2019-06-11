import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PANDAS DATAFRAME
# passing in a dictionary to create a dataframe.
# each list is a column
# number of elements in each list equal number rows
df = pd.DataFrame({'Column1': [1, 2, 3, 4], 'Column2':  [5, 7, 9, 1], 'Column3': [10, 11, 15, 0]}, index=list(range(4)))
print(df)

# getting the module name of te Dataframe class
print(pd.DataFrame.__module__)  # this comes from pandas.core.frame module
# getting the doc string
print(pd.DataFrame.__doc__)
print(df)
print(df.index)

print('')

# Using numpy array dataframe
# two lists, so two rows
# five elemetns in each list so five columns
# shape is 2 * 5
df2 = pd.DataFrame(np.array([[1, 2, 3, 4, 5], [7, 9, 10, 11, 13]]), columns=['H', 'O', 'P', 'Q', 'W'], index=['a', 'b'])
print(df2)
print(df2.shape)
df2 = df2.T

# use rename function and pass in a dictionary with old_name and new_name to change df names.
df2.rename(columns={'a': 'Zero', 'b': 'One'}, inplace=True)
print(df2)

# PANDAS SERIES
# Each list inside the outer list is a row, the number of elements in each list are the number of columns.
# Each element in a list has one value.
print('')
s = pd.Series([[1, 2, 3], [3, 2, 7]], index=['p', 'q'])
print(s)

# Indexing Pandas series
print(s[0])
print('')

print(s[0][1])
print(s[1][2])

# Indexing DataFrame
print(df)
print('')
print(df['Column1'])
print('')
print('here we index a particular value')
print(df['Column1'][2])

# can change index, index is an attribute of the dataframe class.
df.index = ['a', 'b', 'c', 'd']

print('')
print(df)
print('')

# Converting dict to series by passing dict into Series constructor.
my_dict = {'Shiva': 25, 'Nandan': 67, 'Jaya': 64}
print(my_dict)
print('')

srs = pd.Series(my_dict)
print(srs)

#  The default sequence is alphabetical order of column names
# can specify a different ordering

# adding an extra column gives it all NaN values
new_frame = pd.DataFrame(df, columns=['Column3', 'Column1', 'Column2'])
print(new_frame)
print('')
print('accessing rows with iloc')
print('')

# iloc is integer based indexing.
print(new_frame.iloc[2])

# New column which has been passed when the df is constructed contains only Nan's
# we can populate them by accessing the column with bracket notation and assigning
# a desired value
t = pd.DataFrame(df, columns=['Column2', 'Column1', 'Column3', 'New_column'])
print('')
print('')
t['New_column'] = 1  # this is inplace

# modifying a field that does not exist in the dataFrame will create that and
#  assign the value you have defined for that particular field.
t['Col5'] = 'hai'
print(t)
print('')
print('Using loc to get row a')
print('')

print(t.loc['a'])  # need not specify any values or even : in the column dimension

print('')

print('')
print('Using loc to get Column3')
print(t.loc[:, 'Column3'])  # need to specify : in the row dimension to get all rows, then pass in desired Column.
print('')
print('with bracket notation')
print(t['Column2']) # retunrs series




print('')
print('Can specify both row index and Column Index')
print(t.loc['b', 'Column3'])  # this gives just a single value

# if you using bracket notation, Column comes first
print(t['Col5']['a'])

print('')
print('we can tranppose a dataframe')

t = t.T
print(t)


# Reindexing Series and dataframes

# Reindexing is the act of creating a new index of your dataframe from a previously existing
# dataframe but changing the ordering of your index, hence re-indexing
print('')

S = pd.Series([500, 700, 900], index=['a', 'b', 'c'])
print(S)


G = S.reindex(['b', 'a', 'c'])
print(G)  # the values don't change, the order does
print('')

print(df2)

f = df2.reindex(['O', 'P', 'Q', 'W', 'H'])
print(f)

fields = ['One', 'Zero']

f = f.reindex(columns = fields)
print(f)

# WE can subset multiple rows and columns as well

print(f.loc[['O', 'Q'], 'Zero'])

print(f.loc['W', :])

print('')

d = t.T
print(d)

print(d.loc[:,'Column1':'Col5'])  # this includes the endpoint
print(d.loc['a':'b', :])
print(d.loc['a':'b', 'Column3':'New_column'])  # slice of row and slice of column
print('')

# deleting rows and columns from a dataframe

print(d)

print(d.drop(['a']))
print('')
print(d.drop('b'))
print('')

print(d.drop(['b', 'c']))

# for column we need to specify the axis
print(d.drop('Column1', axis=1))  # this is called Column1

# Arithmetic Operations on dataframe and series

a = pd.Series([1, 2, 3])

b = pd.Series([10, 12, 31, 20])

# the extra index is repersented as a None
print(a + b)  # add each index of the series together,

print(2 + a)  # broadcasting, here a number is converted into a series and then added together.

end = {'age': [25, 26, 27], "salary": [50000, 75000, 100000]}
print(end)

# number of elements inside the list is equal to the number of rows, thus the indices should also be the same.
L = pd.DataFrame(end, index=['person1', 'person2', 'person3'])
print(L)

# one way to reverse the index of the dataframe
L = L.reindex(list(reversed(list(L.index))))

print(L)

end2 = {'age': [29, 21, 90], "salary": [50000, 75000, 100000], "weigth": [100, 75, 67]}

R = pd.DataFrame(end2, index=['person1', 'person2', 'person3'])
print(R)

print(L + R)

# you cannot add two add dataframes if there indices are not same.
# also the extra column in any datframe will give us a default 1.

# Arithemtic operations between dataframes and serieis

# You will get all Nan's if you add a series without and index to a dataframe, you will all Nan's
# the index of the seiries is added to each column of your dataframe, the indidces of your dataframe
# are reproduced.

p = pd.Series([2, 19, 20], index=['Stock1', 'Stock2',  "Stock3"])
print(p)

frame = pd.DataFrame({'Stock1': [2, 10, 19], 'Stock2': [5, 15, 10], 'Stock3': [10, 12, 13]}, index=['a', 'b', 'c'])
print(frame)

print(p + frame)


# Sorting Series and Dataframes
print('')

k = pd.Series([10, 20, 11], index=[2, 1, 0])
print(k)

print(k.sort_index())

# Dataframe is sorted when you create it, but need not be the case when you bull

print(frame)

g = pd.DataFrame({'Weight': [5, 1], "height": [10, 15], 'age': [12, 13]})
print(g)

g = g.reindex(columns=['age', 'Weight', 'height'])
print(g)

g = g.sort_index(axis=1)
print(g)
print('')

# Handling duplicate values

# sort the actual value of data-frame of a column instead of sorting index.

print(g.sort_values('Weight'))

print('')
ed = pd.Series([10, 3, 11], index=[5, 10, 1])
print(ed)
# print(ed.sort_index())
# this sorts the actual values of a dataframe.
print(ed.sort_values())

# The index values of serieis and dataframe must always be unique. Duplicate index values must be apprporiately handled

print(ed.index.is_unique)

# Calculating min, max and sum

print(L)

print(L['age'].sum())

print(L.sum())

# calculate sumof rows
print(L.sum(axis=1))
print('')
print(L.idxmax()) # retunrs column and the index at which that columns has lowest value

print(L.idxmin()) # retunrs column and the index at which that columns has lowest value
print('')
print('')
# Dropping nan values

S = pd.Series([1, 7, 10, np.nan], index=[1, 2, 3, 4])
print(S)
print(S.isna())
# print(S.fillna(3)) # CAN FILL NA WITH VALUE OF YOUR CHOICE

# S = S.dropna() # will drop the row with Nan value.
# print(S)

data = pd.read_csv('data/products.csv')
print(data)


data = data.drop("Unnamed: 0", axis=1)
print(data)

# drop row by index value

data = data.drop([2])
print(data)  # the resulting index values don't change

# Arithmetic operations on dataframes

print(data.sort_values(by='product_price'))  # the entire data frame get's sorted according to that column name

# can add up all numeric values in dataframes

print(data.sum(numeric_only=True))

# gives n elements between start and end that are evenly spaced
print(np.linspace(2, 10, 15))

# Logspace, same as linspace, except the intervals are in log.

print(np.logspace(2, 10, 5, endpoint=False))


d = np.arange(10)
print(d)

s = slice(3, 9, 2)
print(s)

print(d[s])
print(d[2:9:3])


# Advanced Indexing and slicing techniques

# Integer indexing

Num = np.arange(24).reshape(6, 4)
print(Num)

print(Num[0:3, 2])

print(Num[:, 2:])

print(Num[0:3, 0:3])

z = Num[[2, 3, 1], [1, 2, 0]]  # this lets us pick individual elements,
# unlike using slicing which gives us rows and values

print(z)


# Boolean indexing

h = Num[Num > 5]  # just Num > 5 returns a boolean value of True where Numhas values greater
# than 5 and false where it doesn't. We use that as a mask on the original array to retireive the actual values
# for which the boolean expression returns True.

print(h)
print('')

# Broadcasting
e = np.array([[1, 2, 3], [2, 10, 11], [4, 5, 7]])

print(e)
print('')
v = np.array([1, 2,3])
print(v)
print('')

print(e+v)
print('iterating')
print('')
for x in np.nditer(e):
    print(x)


# x = np.arange(1, 11)
# print(x)
# y = 2 * x**2 + 700.9

# plt.title('Y=mx +b')
# plt.plot(x, y)
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.plot()
# plt.show()


# we can import flat files with pandas
# and numpy.

data_pd = pd.read_csv('data/products.csv')
print(data_pd)

# drop column mentiond axis = 1
data_pd = data_pd.drop("Unnamed: 0", axis=1)

print(data_pd)

# drop many columns in a list
data_pd = data_pd.drop(['product_name', 'product_category', 'brand'], axis=1)

# use np.loadtxt function to import txt data from Python

# load text data with numpy, use dilimeter to mention seperator,
# skiprows to skip a row.
t = np.loadtxt('new_data', delimiter=" ", skiprows=1)
print(t)

# this is the subset t,
# 3rd row, and all column from the first column
get_sub = t[2, 1:]
print(get_sub)
print(get_sub.ndim)

# reshape function takes and array, t here,
# and the row and column dimensions to be passed
# in as a tuple to reshape it
t = np.reshape(t, (get_sub.ndim, 20))  # 1, 20
print(t)


# # can also import string data, not just numeric, specify dtype = str
# with use_cols we can choose the columns to load
f = np.loadtxt('new_data', delimiter=" ", dtype=str, usecols=[0, 2])
# print(f)
# print(f.shape)

# if the delimeter is space
p = np.loadtxt("new_data", delimiter=" ", dtype=str)
print(p)

# gives first five rows
print(data_pd[0:5])


# has the same effect
print(data_pd.head())

# check shape
print(data_pd.shape)

# this gives us the underlying numpy array
print(data_pd.values)
print(data_pd.values.shape)
# both are of the same shape

# loads nrows specified
s = pd.read_csv("data/products.csv", nrows=3)
print(s)

s = list(range(20))
print(s)


for x in s:
    if x%2 == 0:
        print(f"{x} :This is an even number")
    elif x % 2 == 0 and x % 4 == 0:
        print(f'{x} this number is divisibe by 6')
    else:
        print(f'{x}this number is an odd number')


X

