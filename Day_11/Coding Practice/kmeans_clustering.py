import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


def get_dataset():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df, iris.target_names


def processing_data(df):
    # Standardize the features
    scaler = StandardScaler()
    features = df.drop("target", axis=1)
    scaled_features = scaler.fit_transform(features)
    return scaled_features


def run_kmeans_clustering(scaled_features):
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_features)
        inertia = kmeans.inertia_
        wcss.append(inertia)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.savefig("../graphs/elbow_method.png")
    plt.show()

    return wcss


def finalize_kmeans(scaled_features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_features)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    print(f"Cluster Centers:\n{centers}")
    print(f"Labels:\n{labels}")

    return labels, centers


def cluster_graph(df, labels):
    df["cluster"] = labels

    print("Columns after adding cluster:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head())

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x=df.columns[0],
        y=df.columns[1],
        hue="cluster",
        palette="viridis"
    )

    plt.title("KMeans Clustering")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.legend(title="Cluster")

    plt.savefig("../graphs/kmeans_clusters.png")
    plt.show()


def main():
    df, target_names = get_dataset()

    scaled_features = processing_data(df)

    wcss = run_kmeans_clustering(scaled_features)

    labels, centers = finalize_kmeans(scaled_features, n_clusters=3)

    cluster_graph(df, labels)


if __name__ == "__main__":
    main()