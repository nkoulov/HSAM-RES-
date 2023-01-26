import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("/Users/nicole/Desktop/TVB/HSAM/s123229_fc_TVBSchaeferTian420.txt", sep=" ", header=None)
print(df)

# sns.set()

# df = sns.load_dataset()

# mask = np.triu(np.ones_like(df.corr()))

# dataplot = sns.heatmap(df.corr(), mask=mask)


# plt.show()


###analysis

fc = df

node = [408, 418, 409, 410, 419, 420, 403, 413, 402, 412, 401, 411, 407, 417, 404, 414, 406, 416, 405, 415]
node = [x -1 for x in node]

nmx = fc.iloc[node,node]

print(nmx)


dataplotsubcort = sns.heatmap(fc.iloc[node, node])

# maskagain = np.triu(np.ones_like(dataplotsubcort.corr()))

plt.show()

####FC stat generator

# stat
# preprocessing

FC_tria = np.tril(nmx, k=0)
ff = FC_tria.flatten(order='A')
fc_ff = [f for f in ff if f < 1 and f > -1 ]
plt.show()


#preprocessing

fc_np = nmx.to_numpy()
fc_flatten = fc_np[np.triu_indices(len(nmx.columns))]
fc_flatten = [f for f in fc_flatten if f < 1]




# mean

mean_fc = np.mean(fc_flatten)
print(f"The mean of the FC is {mean_fc}")


# histogram
fig = plt.figure(figsize=(5,5),dpi = 300)
axes = fig.add_subplot(111)
axes.set_title("The Histogram of the FC subcortical distribution")
sns.histplot(fc_flatten, ax=axes)
plt.show()
