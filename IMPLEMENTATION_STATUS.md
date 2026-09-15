# Implementation Status

This document separates the behavior of the current public repository from recovered project context and unrecovered historical details.

## Implemented in the public repository

- Read `.xlsx` or `.xlsm` workbooks with `openpyxl` through pandas.
- Select participant sheets matching the configured `A_#`, `B_#`, or `C_#` naming pattern.
- Parse repeated Monday–Sunday four-column activity blocks.
- Normalize clock values supplied as Excel day fractions, Python time/datetime objects, pandas timestamps, or parseable time strings.
- Forward-fill contextual diary cells while never forward-filling timestamps.
- Reconstruct event end times from the next event's start time and apply a configurable terminal duration to the last event of each participant-day.
- Categorize activity text with explicit keyword rules.
- Build participant-day behavioral duration/count features while keeping daily mean mood outside the clustering feature set.
- Standardize numeric behavioral features and run HDBSCAN when the number of participant-days is sufficient.
- Report HDBSCAN cluster count, noise count/fraction, and mean assignment probability.
- Build TF-IDF unigram/bigram representations.
- Optionally create normalized Sentence-BERT embeddings with `all-MiniLM-L6-v2` and compute within-participant weekday/weekend cosine distance.
- Compare textual routine descriptions with normalized `SequenceMatcher` similarity.
- Compute descriptive Pearson and Spearman associations between behavioral features and daily mean mood.
- Optionally produce a deterministic two-dimensional t-SNE projection for visualization.
- Export event-level data, daily features, clustering results, mood summaries, correlations, quality diagnostics, and a run manifest.
- Validate configuration, parsing, feature engineering, statistics, categorization, and selected semantic utilities with automated tests.
- Run CI across Python 3.11, 3.12, and 3.13, plus Ruff linting, dependency auditing, and a full runtime import-compatibility gate.

## Important interpretation boundary

Mood is retained as an interpretation/outcome variable. It is deliberately excluded from the behavioral clustering feature set. This prevents cluster construction from directly encoding the variable later used to interpret the clusters.

Pearson/Spearman outputs are descriptive associations only. The current implementation does not fit a repeated-measures, hierarchical, causal, or longitudinal model, so participant-day rows must not be treated as independent evidence of causation.

## Deliberately not claimed

- The current modules are not represented as literal recovered historical source code.
- The exact historical row count is not claimed without the original source workbook.
- Exact historical HDBSCAN hyperparameters, final cluster counts, and historical metric values are not presented as verified results.
- t-SNE coordinates are not treated as clusters or as evidence of cluster quality.
- The public repository does not contain the authentic student diary workbook.
- The public repository does not claim clinical, psychological, or causal interpretation of mood.
- The keyword activity taxonomy is a transparent heuristic, not a validated behavioral ontology.
- Sentence-BERT assets are retrieved from the configured upstream model repository on first use unless already cached; the upstream model revision is not pinned byte-for-byte.

## Appropriate future extensions

Potential research/production extensions include participant-aware statistical models, sensitivity analysis for HDBSCAN parameters, cluster-stability assessment, validated activity ontologies, explicit missing-data analysis, model/revision pinning, longitudinal mixed-effects models, and privacy-reviewed de-identified demonstration data.
