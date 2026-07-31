import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Reusing your existing logic where it directly applies
from iris_clustering_project import get_dataset, processing_data, finalize_kmeans

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Clustering Explorer", page_icon="🌸", layout="wide")

sns.set_style("whitegrid")
PALETTE = "muted"   # soft, professional colors instead of bright/heavy ones
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 13

st.title("🌸 K-Means Clustering & PCA Explorer")
st.caption("Upload your own dataset or explore the built-in Iris dataset with interactive clustering and PCA.")

# ---------------------------------------------------------------------------
# Sidebar — Data source
# ---------------------------------------------------------------------------
st.sidebar.header("1. Data Source")
data_source = st.sidebar.radio("Choose a dataset", ["Use Iris Dataset (default)", "Upload your own CSV"])

uploaded_file = None
if data_source == "Upload your own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    target_names = None
    is_iris = False
elif data_source == "Use Iris Dataset (default)":
    df, target_names = get_dataset()
    is_iris = True
else:
    df, target_names = None, None
    is_iris = False

if df is None:
    st.info("Upload a CSV file from the sidebar to get started, or switch to the Iris dataset.")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()

# ---------------------------------------------------------------------------
# Sidebar — Feature / label selection (only meaningful for custom CSVs)
# ---------------------------------------------------------------------------
st.sidebar.header("2. Feature Selection")
if is_iris:
    feature_cols = [c for c in df.columns if c != "target"]
    label_col = "target"
    st.sidebar.caption("Using all 4 Iris measurements as features (target column excluded).")
else:
    feature_cols = st.sidebar.multiselect(
        "Select numeric feature columns to cluster on",
        options=numeric_cols,
        default=numeric_cols,
    )
    label_options = ["None"] + [c for c in df.columns if c not in feature_cols]
    label_choice = st.sidebar.selectbox("Optional label column (for coloring only)", options=label_options)
    label_col = None if label_choice == "None" else label_choice

if len(feature_cols) < 2:
    st.warning("Please select at least 2 numeric feature columns from the sidebar.")
    st.stop()

st.sidebar.header("3. Clustering Settings")
n_clusters = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=3)
show_raw_data = st.sidebar.checkbox("Show raw dataset", value=False)

# ---------------------------------------------------------------------------
# Dataset Preview & Shape
# ---------------------------------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Features Selected", len(feature_cols))

if show_raw_data:
    st.dataframe(df, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Scale features — reuse processing_data() for Iris, generic scaler otherwise
# ---------------------------------------------------------------------------
if is_iris:
    scaled_features = processing_data(df)
else:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])

st.subheader("Dimensions Before PCA")
st.write(f"Scaled feature matrix shape: **{scaled_features.shape[0]} rows × {scaled_features.shape[1]} dimensions**")

st.divider()

# ---------------------------------------------------------------------------
# Apply K-Means (with Elbow Method)
# ---------------------------------------------------------------------------
st.subheader("Apply K-Means Clustering")

st.markdown("**Elbow Method** — helps identify a good value for *k* before committing to one.")


@st.cache_data
def compute_wcss(_scaled_features):
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(_scaled_features)
        wcss.append(km.inertia_)
    return wcss


wcss = compute_wcss(scaled_features)

fig_elbow, ax = plt.subplots(figsize=(5, 3))
ax.plot(range(1, 11), wcss, marker="o", color="#4C72B0")
ax.axvline(n_clusters, color="#DD8452", linestyle="--", label=f"Selected k = {n_clusters}")
ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("WCSS")
ax.legend()

elbow_col, _ = st.columns([2, 1])
with elbow_col:
    st.pyplot(fig_elbow)

apply_kmeans = st.button("Apply K-Means", type="primary")

if apply_kmeans or "labels" in st.session_state:
    if apply_kmeans:
        labels = finalize_kmeans(scaled_features, n_clusters=n_clusters)
        st.session_state["labels"] = labels
        st.session_state["k_used"] = n_clusters
    labels = st.session_state["labels"]

    if st.session_state.get("k_used") != n_clusters:
        st.info("Cluster count changed — click **Apply K-Means** again to refresh results.")

    st.success(f"K-Means applied with k = {st.session_state.get('k_used')}")

    clustered_df = df.copy()
    clustered_df["cluster"] = labels

    # ---- Original vs Clustered (2D, first two selected features) ----
    st.markdown("**Original Data vs. K-Means Clusters**")
    c1, c2 = st.columns(2)
    x_feat, y_feat = feature_cols[0], feature_cols[1]

    with c1:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        hue_col = label_col if label_col else None
        sns.scatterplot(data=df, x=x_feat, y=y_feat, hue=hue_col, palette=PALETTE, ax=ax1)
        ax1.set_title("Original Data")
        st.pyplot(fig1)

    with c2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=clustered_df, x=x_feat, y=y_feat, hue="cluster", palette=PALETTE, ax=ax2)
        ax2.set_title("K-Means Clusters")
        st.pyplot(fig2)
else:
    st.info("Click **Apply K-Means** to run clustering.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# PCA (used for both the 3D view and the 2D view — computed once)
# ---------------------------------------------------------------------------
st.subheader("Apply PCA")

n_pca_components = min(3, scaled_features.shape[1])
apply_pca = st.button("▶ Apply PCA", type="primary")

if apply_pca or "pca_df" in st.session_state:
    if apply_pca:
        pca = PCA(n_components=n_pca_components)
        pcs = pca.fit_transform(scaled_features)
        pca_cols = [f"PC{i+1}" for i in range(n_pca_components)]
        pca_df = pd.DataFrame(pcs, columns=pca_cols)
        pca_df["cluster"] = labels
        if label_col:
            pca_df["label"] = df[label_col].values
        st.session_state["pca_df"] = pca_df
        st.session_state["explained_variance"] = pca.explained_variance_ratio_

    pca_df = st.session_state["pca_df"]
    explained_variance = st.session_state["explained_variance"]

    st.subheader("Dimensions After PCA")
    st.write(
        f"Reduced from **{scaled_features.shape[1]} dimensions** to "
        f"**{n_pca_components} principal components**, "
        f"explaining **{explained_variance.sum() * 100:.1f}%** of total variance."
    )

    # ---- 2D PCA Visualization ----
    st.subheader("PCA Visualization (2D)")
    hue_source = "label" if label_col and "label" in pca_df.columns else "cluster"
    fig_pca, ax3 = plt.subplots(figsize=(5, 3.5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=hue_source, palette=PALETTE, ax=ax3)
    ax3.set_title("PCA Visualization")

    pca_col, _ = st.columns([2, 1])
    with pca_col:
        st.pyplot(fig_pca)

    # ---- 3D Cluster Visualization ----
    st.subheader("🧊 3D Cluster Visualization")
    if n_pca_components >= 3:
        fig_3d = px.scatter_3d(
            pca_df,
            x="PC1", y="PC2", z="PC3",
            color=pca_df["cluster"].astype(str),
            opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Clusters in 3D PCA Space",
        )
        fig_3d.update_layout(legend_title_text="Cluster", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("Need at least 3 features to render a 3D view.")

    st.divider()

    # ---- Show Cluster Labels ----
    st.subheader("Cluster Labels")
    result_df = df.copy()
    result_df["cluster"] = labels

    counts = result_df["cluster"].value_counts().sort_index()
    st.bar_chart(counts)

    st.dataframe(result_df, use_container_width=True)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download data with cluster labels", data=csv_bytes, file_name="clustered_data.csv", mime="text/csv")

else:
    st.info("Click **Apply PCA** to see dimensionality reduction and 3D visualization.")