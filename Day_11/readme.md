# Day 11 - Unsupervised Learning: Clustering & PCA

## Overview

In this task, I explored **Unsupervised Learning** using the Iris dataset from Scikit-learn. Unlike supervised learning, unsupervised learning works without target labels and is used to discover hidden patterns in data.

The project includes:

- Dataset Exploration
- K-Means Clustering
- Elbow Method
- Principal Component Analysis (PCA)
- Data Visualization
- Mini Project

---

## What is Clustering?

Clustering is an **unsupervised learning technique** that groups similar data points into clusters based on their feature values. The goal is to place similar observations in the same cluster while separating dissimilar ones.

In this project, I used the **K-Means Clustering** algorithm to group Iris flower samples into clusters.

---

## What is PCA?

**Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms high-dimensional data into fewer dimensions while preserving as much information as possible.

In this project, PCA reduced the original **4 features** into **2 principal components**, making the dataset much easier to visualize.

---

## How did I determine the best value of K?

I used the **Elbow Method** to determine the optimal number of clusters.

Steps:

1. Trained K-Means models with different values of **K (1 to 10)**.
2. Calculated the **WCSS (Within Cluster Sum of Squares)** for each value.
3. Plotted the Elbow graph.
4. Selected the point where the decrease in WCSS started slowing down.

The elbow appeared at:

**K = 3**

This matches the three species present in the Iris dataset.

---

## Project Workflow

1. Load Iris Dataset
2. Explore Dataset
3. Standardize Features
4. Apply K-Means Clustering
5. Find Best K using Elbow Method
6. Visualize Clusters
7. Apply PCA
8. Visualize PCA Results
9. Analyze Observations

---

## Insights from the Visualizations

- The Elbow Method indicated that **3 clusters** were the optimal choice.
- K-Means successfully grouped most Iris samples into three meaningful clusters.
- The Setosa species was clearly separated from the other two species.
- Versicolor and Virginica showed slight overlap, which is expected because their feature values are similar.
- PCA reduced the dataset from **4 dimensions to 2 dimensions** while preserving most of the important information.
- The PCA visualization made the clusters much easier to understand and interpret.

---

## Files Included

- `dataset_exploration.py`
- `kmeans_clustering.py`
- `pca.py`
- `iris_clustering_project.py`
- `elbow_method.png`
- `kmeans_clusters.png`
- `pca_visualization.png`
- `README.md`

---

## Conclusion

This task helped me understand the fundamentals of **Unsupervised Learning**, how **K-Means Clustering** groups similar data points, how the **Elbow Method** helps choose the optimal number of clusters, and how **PCA** simplifies high-dimensional data for better visualization and interpretation.