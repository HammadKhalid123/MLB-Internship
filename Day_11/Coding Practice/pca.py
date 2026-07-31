from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def get_dataset():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df, iris.target_names


def pca_analysis(df):
    # Standardize the features
    features = df.drop("target", axis=1)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)

    # Create PCA DataFrame
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=["PC1", "PC2"]
    )

    pca_df["target"] = df["target"]

    return pca_df, pca.explained_variance_ratio_


def plot_pca(pca_df):
    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="target",
        palette="Set1",
        s=100
    )

    plt.title("PCA of Iris Dataset")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Target")
    plt.grid(True)

    plt.savefig("pca_visualization.png")
    plt.show()


def main():
    df, target_names = get_dataset()

    pca_df, explained_variance_ratio = pca_analysis(df)

    print("Explained Variance Ratio:")
    print(explained_variance_ratio)

    plot_pca(pca_df)


if __name__ == "__main__":
    main()