from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

linked = linkage(X, method='ward')

plt.figure(figsize=(6,4))
dendrogram(linked)
plt.title("Dendrogram")
plt.show()

# Apply hierarchical clustering
hc = AgglomerativeClustering(n_clusters=2)
labels = hc.fit_predict(X)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title("Hierarchical Clustering")
plt.show()
