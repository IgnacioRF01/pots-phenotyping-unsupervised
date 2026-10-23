"""
Generate a synthetic dataset with the same 13 variables used in the original
(unpublished) exploratory analysis, so the pipeline can be run end to end
without any patient data.

The values are SIMULATED. They follow three latent groups (healthy controls and
two dysautonomia-like phenotypes) with plausible magnitudes, but they are not
derived from, and do not reproduce, any real patient.

Usage:
    python src/synthetic_data.py --n 60 --out data/synthetic_dataset.csv
"""
import argparse

import numpy as np
import pandas as pd

FEATURES = [
    "Theta SUP", "Theta ORTHO", "Alfa ORT", "CompassPlus", "Delta Sys",
    "GAMA Zero", "Delta HR", "D Labilidad-E", "A/B Index ORTHO",
    "A/B Index SUP", "Delta Theta", "Area Orbita ORTHO", "Gamma RISE",
]

# Group means (control, phenotype 1, phenotype 2) and a common SD per variable.
# Chosen so that baroreflex-related variables (Theta, A/B index, ellipse area)
# separate the groups, as observed qualitatively in the original exploration.
PROFILE = {
    #                      control  pheno1  pheno2   sd
    "Theta SUP":         (  45.0,   30.0,   22.0,   6.0),
    "Theta ORTHO":       (  32.0,   18.0,   12.0,   6.0),
    "Alfa ORT":          (   1.0,    0.7,    0.5,   0.2),
    "CompassPlus":       (  10.0,   45.0,   60.0,  10.0),
    "Delta Sys":         (   5.0,   15.0,   25.0,   6.0),
    "GAMA Zero":         (   0.0,    1.5,    2.5,   1.0),
    "Delta HR":          (  15.0,   35.0,   28.0,   7.0),
    "D Labilidad-E":     (   0.3,    0.6,    0.8,   0.2),
    "A/B Index ORTHO":   (   2.0,    4.0,    5.5,   0.8),
    "A/B Index SUP":     (   1.8,    3.2,    4.5,   0.7),
    "Delta Theta":       ( -13.0,  -12.0,  -10.0,   5.0),
    "Area Orbita ORTHO": (2500.0, 1500.0, 1000.0, 400.0),
    "Gamma RISE":        (   1.0,    1.3,    1.6,   0.4),
}
GROUP_NAMES = ["control", "phenotype_1", "phenotype_2"]


def make_dataset(n=60, proportions=(0.35, 0.35, 0.30), seed=42):
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, proportions)
    rows, labels = [], []
    for g, count in enumerate(counts):
        block = {
            name: rng.normal(PROFILE[name][g], PROFILE[name][3], count)
            for name in FEATURES
        }
        rows.append(pd.DataFrame(block))
        labels += [GROUP_NAMES[g]] * count
    df = pd.concat(rows, ignore_index=True)
    df["true_group"] = labels  # kept only to evaluate the clustering
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", [f"S{i + 1:03d}" for i in range(len(df))])
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=60, help="number of synthetic subjects")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/synthetic_dataset.csv")
    args = parser.parse_args()

    df = make_dataset(n=args.n, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} synthetic subjects to {args.out}")
    print(df["true_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
