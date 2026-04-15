# ===============================
# Task-02: K-Means Clustering
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -------------------------------
# Load LOCAL Dataset
# -------------------------------
data = pd.read_csv("Mall_Customers (1).csv")   # 👈 change file name if needed

print("Dataset Loaded Successfully!\n")
print(data.head())

# -------------------------------
# Check Columns
# -------------------------------
print("\nColumns in dataset:")
print(data.columns)

# -------------------------------
# Select Features
# -------------------------------
# Make sure these column names match your dataset
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# -------------------------------
# Elbow Method
# -------------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure()
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# -------------------------------
# Apply K-Means (K = 5)
# -------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# -------------------------------
# Plot Clusters
# -------------------------------
plt.figure()

for i in range(5):
    plt.scatter(
        X.iloc[y_kmeans == i, 0],
        X.iloc[y_kmeans == i, 1],
        label=f'Cluster {i+1}'
    )

# Plot Centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X',
    label='Centroids'
)

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

# -------------------------------
# Print Cluster Centers
# -------------------------------
print("\nCluster Centers:\n", kmeans.cluster_centers_)