"""
Exploratory alternative: Gaussian Mixture Model (GMM) clustering.

Two variants were explored during the original analysis:
  * GMM on a reduced set of standardized variables (default: first 4)
  * GMM on PCA-transformed data (--use-pca)

Usage:
    python src/gmm_exploration.py --data data/synthetic_dataset.csv
    python src/gmm_exploration.py --data data/synthetic_dataset.csv --use-pca --n-components 4
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

META_COLUMNS = {"id", "ID", "true_group"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/synthetic_dataset.csv")
    parser.add_argument("--out", default="results")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--n-features", type=int, default=4,
                        help="number of variables used without PCA (default 4)")
    parser.add_argument("--use-pca", action="store_true", help="cluster on PCA components")
    parser.add_argument("--n-components", type=int, default=4, help="PCA components (with --use-pca)")
    parser.add_argument("--k", type=int, default=3, help="number of mixture components")
    args = parser.parse_args()

    matplotlib.use("Agg")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    raw = pd.read_csv(args.data)
    meta = raw[[c for c in raw.columns if c in META_COLUMNS]]
    features = raw[[c for c in raw.columns if c not in META_COLUMNS]]

    if args.use_pca:
        X = StandardScaler().fit_transform(features.iloc[:, :13])
        X = PCA(n_components=args.n_components).fit_transform(X)
        axis_names = ("PC1", "PC2")
        tag = f"pca{args.n_components}"
    else:
        features = features.iloc[:, :args.n_features]
        X = StandardScaler().fit_transform(features)
        axis_names = (f"{features.columns[0]} (z)", f"{features.columns[1]} (z)")
        tag = f"{args.n_features}vars"

    gmm = GaussianMixture(n_components=args.k, random_state=42).fit(X)
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X).max(axis=1)

    out = meta.copy()
    out["cluster"] = labels
    out["membership_probability"] = probs
    out.to_csv(os.path.join(args.out, f"gmm_assignments_{tag}.csv"), index=False)

    print(f"GMM ({args.k} components, {tag}): BIC = {gmm.bic(X):.1f}")
    if "true_group" in meta:
        print(f"Adjusted Rand index vs synthetic ground truth: "
              f"{adjusted_rand_score(meta['true_group'], labels):.3f}")

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.8,
                     edgecolor="k", linewidth=0.3)
    plt.colorbar(sc, label="Cluster", ticks=sorted(set(labels)))
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.title(f"Gaussian Mixture Model clustering ({tag})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(args.figures, f"08_gmm_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Figure saved to {path}")


if __name__ == "__main__":
    main()
