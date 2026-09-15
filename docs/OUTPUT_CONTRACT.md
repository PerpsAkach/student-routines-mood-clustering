# Output Contract

The runner writes outputs to `--output-dir` (default: `outputs`). Files are overwritten on a new run using the same directory.

## Always written

### `events_clean.csv`

Normalized event-level records, including participant ID, day, normalized start/end times, reconstructed duration, activity text, contextual fields, numeric satisfaction when parseable, and the rule-based activity category.

### `daily_features.csv`

One row per participant/day. Behavioral columns include category-duration totals plus event count, total observed minutes, and a weekend indicator. `Mean_Mood` is retained for interpretation but is excluded from the clustering feature selection helper.

### `daily_clusters.csv`

The daily feature table plus `Cluster` and `Cluster_Probability` when HDBSCAN runs. If there are fewer participant-days than `hdbscan_min_cluster_size`, clustering is skipped explicitly and these fields remain missing.

### `mood_by_category.csv`

Descriptive category-level summaries:

- mean numeric satisfaction;
- median numeric satisfaction;
- count of non-missing numeric satisfaction values;
- total reconstructed minutes.

### `mood_feature_correlations.csv`

Descriptive Pearson and Spearman associations between behavioral participant-day features and `Mean_Mood`. Complete finite pairs are used. Results with insufficient observations or constant variables remain missing rather than manufacturing a statistic.

### `data_quality.json`

Run-level input diagnostics, including event/participant counts, participant-days, repeated timestamps, zero-duration rows, missing satisfaction/activity values, and invalid durations.

### `clustering_diagnostics.json`

Contains an explicit status. When clustering completes, diagnostics include:

- observation count;
- number of substantive clusters (excluding HDBSCAN label `-1`);
- number/fraction of HDBSCAN noise assignments;
- mean assignment probability.

When clustering cannot run because the input is too small or lacks usable numeric behavioral features, the file records the skip reason.

### `run_manifest.json`

Compact summary of the run, including event/participant/day counts plus quality, clustering, semantic-analysis, and t-SNE status blocks.

## Optional: `--semantic`

### `weekday_weekend_semantic_distance.csv`

For participants with both weekday and weekend activity text, the pipeline creates normalized Sentence-BERT embeddings and writes cosine distance:

```text
Semantic_Distance = 1 - cosine_similarity(weekday_embedding, weekend_embedding)
```

The theoretical range is clipped to `[0, 2]` to protect against floating-point overshoot.

## Optional: `--tsne`

### `tsne_projection.csv`

Two-dimensional t-SNE coordinates joined to participant/day identifiers.

### `tsne_projection.png`

A scatter plot of the t-SNE projection. If cluster labels are available, points are grouped by label for visualization only.

`t-SNE` is not used to fit the HDBSCAN clusters and should not be interpreted as a cluster-quality metric.

## Interpretation rules

- Cluster label `-1` is HDBSCAN noise/outlier assignment, not a substantive routine cluster.
- Cluster numeric labels are arbitrary identifiers and have no ordinal meaning.
- `Cluster_Probability` is HDBSCAN membership strength, not a probability of psychological state or mood.
- Correlation outputs are descriptive association statistics, not causal estimates.
- Generated t-SNE coordinates are stochastic-embedding visualization coordinates with a fixed random seed in this implementation; distances in the 2D plot should not be overinterpreted.
