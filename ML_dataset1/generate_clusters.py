import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

data = np.loadtxt(r"bert_embeddings_plot.txt")
data1 = data[:,1:]


# find optimal k using elbow method
distortions = []
K = range(1,30)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(data1)
    distortions.append(kmeanModel.inertia_)

"""
# plot elbow curve
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
"""
kmeans = KMeans(n_clusters=10)  #n_clusters=7  for book data

labels = kmeans.fit_predict(data1)
centroids = kmeans.cluster_centers_

# calculate distances of data points from centroids
distances = cdist(data1, centroids)

# assign new labels based on minimum distance
new_labels = np.argmin(distances, axis=1)
df = pd.DataFrame({"Cluster_id": new_labels})

data2=data[:, 0].astype(int)
movie_ids = pd.DataFrame(data2, columns=['movie_ids'])

df1 = pd.concat([df, movie_ids], axis=1)

#rating_df = pd.read_csv('ratings.csv')
column_names = ['user_id', 'item_id', 'rating', 'ts']
rating_df = pd.read_csv(r"ratings.dat", sep = "::", names = column_names, engine='python')

merged_df = pd.merge(df1, rating_df, left_on='movie_ids', right_on='item_id')
merged_df = merged_df.drop('movie_ids', axis=1)

merged_df.to_csv('updated_ratingwithcluster_id.csv', index=False)











