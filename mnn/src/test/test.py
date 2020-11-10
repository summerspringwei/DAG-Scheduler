from sklearn.cluster import KMeans
import numpy as np
X = np.array([[32], [32], [32], [16], [16], [16], [16], \
    [16], [32], [8], [8], [8], [8], [4], [4], \
        [8], [8], [8], [4], [4]])
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print(kmeans.labels_)
