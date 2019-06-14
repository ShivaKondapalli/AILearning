import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

file_path = 'data/flights.csv'

df = pd.read_csv(file_path).drop('Unnamed: 0', axis=1)
print(df.columns)

print(df[['year', 'month', 'day']])

print(df.head(15))


# Histogram of Continous data
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
bin_edges = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
ax1.hist(df['dep_delay'], bins=bin_edges)
ax1.set_xlabel('Departure delays of Flights')
ax1.set_ylabel('Number of flights')
ax1.set_title('Distribution of departure delays')

# bee swarm plot of United airlines and arr_delay
ax2 = fig.add_subplot(2, 2, 2)
sub = df[['name', 'dep_delay']].head(50)

_ = sns.swarmplot(x='name', y='dep_delay', data=sub)
_ = ax2.set_xticklabels([''] + sub['name'], rotation=60)
_ = ax2.set_ylabel('Departure delay')

# Empirical Cumulative distribution Function


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y


ax3 = fig.add_subplot(2, 2, 3)

x_arr_delay, y_arr_delay = ecdf(df['arr_delay'])
_ = ax3.plot(x_arr_delay, y_arr_delay, marker='.', linestyle='none')
_ = ax3.set_xlabel('arrival delay')
_ = ax3.set_ylabel('ECDF')

plt.show()

