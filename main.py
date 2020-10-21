# Importing the Required libraries
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import datasets


# Reading the data
iris_df = pd.read_csv("Iris.csv", index_col = 0)
print("Let's see a part of the whole dataset - \n")
iris_df.head()

print ("The info about the datset is as follows - \n")
iris_df.info()

# Plotting the pair plot
sns.pairplot(iris_df, hue = 'Species')
# correlation matrix
sns.heatmap(iris_df.corr())
# Defining 'X'
X = iris_df.iloc[:, [0, 1, 2, 3]].values

# Finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the results onto a line graph,
# Allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within cluster sum of squares
plt.show()
sns.set(rc={'figure.figsize': (5, 5)})

# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# Visualising the clusters - On the first two columns
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],
            s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],
            s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')
# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],
            s = 100, c = 'red', label = 'Centroids')
plt.legend()

sns.set(rc={'figure.figsize':(10,8)})

