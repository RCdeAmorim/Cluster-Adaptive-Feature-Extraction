# Cluster-Adaptive Feature Extraction (CAFE)

MATLAB implementation of **Cluster-Adaptive Feature Extraction (CAFE)**, the method introduced in:

> Renato Cordeiro de Amorim and Vladimir Makarenkov.
> *Cluster-Adaptive Feature Extraction and its Theoretical Foundation with Minkowski Weighted k-Means.*

CAFE runs Minkowski weighted k-means (mwk-means), uses the cluster-specific feature weights it learns to rescale the data within each cluster, and then applies a standard feature extraction method (PCA, NMF or ICA) to the rescaled data. Within each cluster, the rescaling suppresses features with high within-cluster dispersion (e.g. noise) and amplifies informative ones. See the paper for the full method and its theoretical justification.

## Files

| File | Description |
|---|---|
| `mwk_FE.m` | CAFE. This is the function you call. |
| `MWKmeans_fast.m` | Minkowski weighted k-means, called by `mwk_FE`. |
| `RandIndex.m` | Adjusted Rand Index, used by `mwk_FE` to select the final clustering. |
| `RealWorldData/` | The noisy UCI data sets used in the paper's experiments (see below). |

## Requirements

- MATLAB R2016b or later (the paper's experiments used R2026a)
- Statistics and Machine Learning Toolbox (`pca`, `nnmf`, `rica`, `kmeans`)

## Usage

Add the repository folder to your MATLAB path, then call:

```matlab
FE = mwk_FE(Data, K, r, type);
```

**Inputs**

| Argument | Description |
|---|---|
| `Data` | *n × m* data matrix (rows are observations, columns are features). |
| `K` | Number of clusters. |
| `r` | Number of features to extract. |
| `type` | Feature extraction method applied after rescaling: `1` = PCA (CAFE+PCA), `2` = NMF (CAFE+NMF), `3` = ICA (CAFE+ICA). |

**Output**

| Output | Description |
|---|---|
| `FE` | *n × r* matrix of extracted features. |

### Preparing your data

Before calling `mwk_FE`, normalise each feature as in Equation (6) of the paper:

```matlab
Data = (Data - mean(Data)) ./ range(Data);
```

Remove any constant features before normalising, so the denominator is never zero. For `type = 2` (NMF), `mwk_FE` shifts the data to be non-negative internally, so you don't need to do this yourself.

### Example

```matlab
load fisheriris                      % example data shipped with MATLAB
X = (meas - mean(meas)) ./ range(meas);
K = 3;                               % number of clusters
r = floor(size(X, 2) / 2);           % number of features to extract, as in the paper

FE = mwk_FE(X, K, r, 1);             % CAFE+PCA

% Cluster the extracted features, as in the paper
labels = kmeans(FE, K, 'Replicates', 100);
```

### What `mwk_FE` does

1. Runs mwk-means 50 times for each Minkowski exponent p ∈ {1.1, 1.5, 2, 3}, and keeps the run with the lowest objective for each p.
2. Of these four clusterings, selects the one with the highest total Adjusted Rand Index against the other three.
3. Rescales each data point by the feature weights of its cluster: x̃ᵢᵥ = wₗᵥ · xᵢᵥ.
4. Applies the chosen method to the rescaled data:
   - PCA: keeps the leading `r` principal components.
   - NMF: 50 replicates, keeping the best one.
   - ICA: reconstruction ICA (`rica`), 50 restarts, keeping the one with the lowest Frobenius reconstruction error.

Because of step 1, a call runs mwk-means 200 times, so it can take a while on larger data sets.

## Data used in the paper

`RealWorldData/` contains noisy versions of nine UCI data sets: Australian Credit Approval, Balance, Breast Cancer, Car Evaluation, Ecoli, Glass, Ionosphere, Iris and Teaching Assistant. File names end in `_xs_yn.mat`, where:

- `x` is the noise magnitude σ ∈ {1, 2, 5}
- `y` is the noise level η × 10, i.e. `2` = 20% and `4` = 40% of (cluster, feature) pairs corrupted

The paper reports results for σ ∈ {1, 2, 5} and η ∈ {20%, 40%}. Files ending in `_1n` (η = 10%) were generated but not used in the paper.

Each file holds a 2 × 20 cell array `Data`, one column per noisy version:

- `Data{1, i}`: the normalised data matrix of version `i`
- `Data{2, i}`: the ground-truth labels (1 … k)

For example:

```matlab
S = load('RealWorldData/<dataset>_5s_4n.mat');
X = S.Data{1, 1};
y = S.Data{2, 1};
K = max(y);
FE = mwk_FE(X, K, floor(size(X, 2) / 2), 1);
```

## Citation

If you use this code, please cite the paper above.
