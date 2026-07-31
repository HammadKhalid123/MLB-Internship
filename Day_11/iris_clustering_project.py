import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def get_dataset():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df, iris.target_names


def processing_data(df):
    scaler = StandardScaler()
    X = df.drop("target", axis=1)
    scaled_features = scaler.fit_transform(X)
    return scaled_features


def run_kmeans_clustering(scaled_features):
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(range(1,11), wcss, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.savefig("./graphs/elbow_method.png")
    plt.show()


def finalize_kmeans(scaled_features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_features)

    print("Cluster Centers:")
    print(kmeans.cluster_centers_)

    return labels


def original_plot(df):
    plt.figure(figsize=(8,6))

    sns.scatterplot(
        data=df,
        x="sepal length (cm)",
        y="petal length (cm)",
        hue="target",
        palette="Set1"
    )

    plt.title("Original Iris Data")
    plt.savefig("original_data.png")
    plt.show()


def cluster_plot(df, labels):
    cluster_df = df.copy()
    cluster_df["cluster"] = labels

    plt.figure(figsize=(8,6))

    sns.scatterplot(
        data=cluster_df,
        x="sepal length (cm)",
        y="petal length (cm)",
        hue="cluster",
        palette="viridis"
    )

    plt.title("K-Means Clusters")
    plt.savefig("kmeans_clusters.png")
    plt.show()

def pca_analysis(scaled_features, df):
    pca = PCA(n_components=2)

    principal_components = pca.fit_transform(scaled_features)

    pca_df = pd.DataFrame(
        principal_components,
        columns=["PC1", "PC2"]
    )

    pca_df["target"] = df["target"]

    print("\nExplained Variance Ratio:")
    print(pca.explained_variance_ratio_)

    return pca_df

def plot_pca(pca_df):
    plt.figure(figsize=(8,6))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="target",
        palette="Set1"
    )

    plt.title("PCA Visualization")
    plt.savefig("pca_visualization.png")
    plt.show()

def observations():
    print("\nObservations")
    print("- Number of Clusters: 3")
    print("- The clusters closely represent the three Iris flower species.")
    print("- PCA reduced 4 features into 2 principal components.")
    print("- PCA made visualization easier while preserving most of the important information.")


def main():
    df, target_names = get_dataset()
    
    print("First 5 Rows:")
    print(df.head())

    print("\nDataset Shape:", df.shape)
    scaled_features = processing_data(df)

    run_kmeans_clustering(scaled_features)
    labels = finalize_kmeans(scaled_features, n_clusters=3)

    original_plot(df)
    cluster_plot(df, labels)

    pca_df = pca_analysis(scaled_features, df)
    plot_pca(pca_df)
    observations()

if __name__ == "__main__":
    main()