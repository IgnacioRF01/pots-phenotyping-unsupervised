"""
Unsupervised phenotyping pipeline for POTS-like autonomic dysfunction.

Steps
  1. Spearman correlation between clinical variables
  2. Hierarchical clustering of the variables (average linkage, 1 - rho
     distance, optimal leaf ordering): dendrogram and reordered heatmap
  3. Standardization and elbow method (k = 1-10)
  4. PCA: scree plot, cumulative explained variance, loadings
  5. K-means in PCA space, comparison of k = 3-6, final model with k = 3
  6. Cluster scatter plot and PCA biplot

Usage:
    python src/main_pipeline.py --data data/synthetic_dataset.csv --out results
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:  # optional: nicer label placement in the biplot
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

RANDOM_STATE = 32
META_COLUMNS = {"id", "ID", "true_group"}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_features(path, max_features=13):
    raw = pd.read_csv(path)
    meta = raw[[c for c in raw.columns if c in META_COLUMNS]]
    features = raw[[c for c in raw.columns if c not in META_COLUMNS]]
    features = features.iloc[:, :max_features]
    return features, meta


# ----------------------------------------------------------------------------
# Variable structure
# ----------------------------------------------------------------------------
def variable_clustering(corr):
    """Average-linkage clustering of variables on the 1 - rho distance,
    with optimal leaf ordering. Returns the linkage matrix and leaf order."""
    distance = 1.0 - corr.values
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method="average")
    Z = optimal_leaf_ordering(Z, condensed)
    return Z, leaves_list(Z)


def plot_dendrogram(Z, labels, out):
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=list(labels), leaf_rotation=35, leaf_font_size=9)
    plt.title("Hierarchical clustering of clinical variables (1 − Spearman ρ, optimal leaf ordering)")
    plt.ylabel("Distance (1 − ρ)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_correlation(corr, order, out):
    reordered = corr.iloc[order, order]
    plt.figure(figsize=(13, 10))
    sns.heatmap(reordered, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.5, square=True, cbar_kws={"label": "Spearman ρ"})
    plt.title("Spearman correlation matrix (ordered by variable clustering)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# Cluster number and PCA
# ----------------------------------------------------------------------------
def plot_elbow(X, out, k_max=10):
    ks = range(1, k_max + 1)
    inertia = [KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X).inertia_ for k in ks]
    plt.figure(figsize=(7, 5))
    plt.plot(ks, inertia, marker="o", linestyle="--")
    plt.xticks(list(ks))
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.title("Elbow method")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_pca_variance(X, out_scree, out_cumulative):
    full = PCA().fit(X)
    comps = np.arange(1, len(full.explained_variance_) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(comps, full.explained_variance_, marker="o", linestyle="--")
    plt.axhline(1.0, color="grey", linewidth=1, linestyle=":", label="Eigenvalue = 1 (Kaiser)")
    plt.xticks(comps)
    plt.xlabel("Principal component")
    plt.ylabel("Eigenvalue")
    plt.title("Scree plot")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_scree, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(comps, np.cumsum(full.explained_variance_ratio_), marker="o", linestyle="--")
    plt.xticks(comps)
    plt.ylim(0, 1.02)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Cumulative explained variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_cumulative, dpi=150)
    plt.close()


def compare_k(X_pca, k_values, true_labels=None):
    """Silhouette score for each k; if reference labels exist (synthetic data),
    also the adjusted Rand index and whether all controls share one cluster."""
    rows = []
    for k in k_values:
        pred = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_pca)
        row = {"k": k, "silhouette": silhouette_score(X_pca, pred)}
        if true_labels is not None:
            row["adjusted_rand_index"] = adjusted_rand_score(true_labels, pred)
            controls = pred[np.asarray(true_labels) == "control"]
            row["controls_in_single_cluster"] = len(set(controls)) == 1
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Visualization of the final model
# ----------------------------------------------------------------------------
def plot_clusters(X_pca, labels, out):
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.8, edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, label="Cluster", ticks=sorted(set(labels)))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("K-means clusters in PCA space")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_biplot(X_pca, components, labels, feature_names, out, scale_text=1.15):
    xs, ys = X_pca[:, 0], X_pca[:, 1]
    xs = xs / (xs.max() - xs.min())
    ys = ys / (ys.max() - ys.min())
    plt.figure(figsize=(9, 8))
    plt.scatter(xs, ys, c=labels, cmap="viridis", alpha=0.7)
    texts = []
    for i, name in enumerate(feature_names):
        dx, dy = components[0, i], components[1, i]
        plt.arrow(0, 0, dx, dy, color="r", alpha=0.5, head_width=0.01)
        texts.append(plt.text(dx * scale_text, dy * scale_text, name, color="darkgreen",
                              ha="center", va="center", fontsize=8))
    reach = max(np.abs(components[:2]).max() * scale_text, np.abs(xs).max(), np.abs(ys).max()) * 1.1
    plt.xlim(-reach, reach)
    plt.ylim(-reach, reach)
    if adjust_text is not None:
        adjust_text(texts)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA biplot (points: subjects; arrows: variable loadings)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/synthetic_dataset.csv")
    parser.add_argument("--out", default="results")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--n-components", type=int, default=4, help="PCA components used for clustering")
    parser.add_argument("--k", type=int, default=3, help="number of clusters in the final model")
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=6)
    args = parser.parse_args()

    matplotlib.use("Agg")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)
    fig = lambda name: os.path.join(args.figures, name)

    features, meta = load_features(args.data)
    true_labels = meta["true_group"].values if "true_group" in meta else None
    print(f"Loaded {features.shape[0]} subjects x {features.shape[1]} variables")

    # 1-2. Variable structure
    corr = features.corr(method="spearman")
    Z, order = variable_clustering(corr)
    plot_dendrogram(Z, features.columns, fig("01_variable_dendrogram.png"))
    plot_correlation(corr, order, fig("02_spearman_correlation.png"))
    corr.to_csv(os.path.join(args.out, "spearman_correlation.csv"))

    # 3. Standardization and elbow
    X = StandardScaler().fit_transform(features)
    plot_elbow(X, fig("03_elbow.png"))

    # 4. PCA
    plot_pca_variance(X, fig("04_scree.png"), fig("05_cumulative_variance.png"))
    pca = PCA(n_components=args.n_components)
    X_pca = pca.fit_transform(X)
    pcs = [f"PC{i + 1}" for i in range(args.n_components)]
    pd.DataFrame({"component": pcs,
                  "eigenvalue": pca.explained_variance_,
                  "explained_variance_ratio": pca.explained_variance_ratio_}
                 ).to_csv(os.path.join(args.out, "pca_variance.csv"), index=False)
    pd.DataFrame(pca.components_, index=pcs, columns=features.columns
                 ).to_csv(os.path.join(args.out, "pca_loadings.csv"))
    print(f"PCA ({args.n_components} components) explains "
          f"{pca.explained_variance_ratio_.sum():.1%} of the variance")

    # 5. Number of clusters, then final model
    comparison = compare_k(X_pca, range(args.k_min, args.k_max + 1), true_labels)
    comparison.to_csv(os.path.join(args.out, "k_comparison.csv"), index=False)
    print("\nComparison of k:\n" + comparison.to_string(index=False, float_format="%.3f"))

    labels = KMeans(n_clusters=args.k, random_state=RANDOM_STATE, n_init=10).fit_predict(X_pca)
    assignments = meta.copy()
    assignments[pcs] = X_pca
    assignments["cluster"] = labels
    assignments.to_csv(os.path.join(args.out, "cluster_assignments.csv"), index=False)
    if true_labels is not None:
        print("\nClusters vs synthetic ground truth:\n"
              + pd.crosstab(assignments["cluster"], true_labels).to_string())

    # 6. Figures of the final model
    plot_clusters(X_pca, labels, fig("06_kmeans_pca_clusters.png"))
    plot_biplot(X_pca, pca.components_, labels, features.columns, fig("07_pca_biplot.png"))
    print(f"\nFigures saved to {args.figures}/, tables saved to {args.out}/")


if __name__ == "__main__":
    main()
