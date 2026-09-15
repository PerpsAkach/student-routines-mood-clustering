# Provenance

This repository distinguishes recovered project context from code reconstructed for the public portfolio and from later engineering enhancements.

## RECOVERED

Supported by prior project context/artifacts:

- multi-sheet student activity workbook structure;
- repeated Monday–Sunday activity blocks;
- reconstruction of event durations from sequential start times;
- a 30-minute terminal duration in the historical workflow;
- activity-text analysis with TF-IDF;
- Sentence-BERT using `all-MiniLM-L6-v2`;
- HDBSCAN for unsupervised routine clustering;
- t-SNE for visualization;
- Pearson/Spearman association analysis;
- weekday/weekend semantic comparison;
- a historical routine-similarity workflow using `SequenceMatcher` with an approximately 85% threshold.

These items describe the recovered methodology/context. They do not imply that every current line of code is original historical source.

## RECONSTRUCTED

The current public implementation was rebuilt from the supported project methodology where literal original source bytes were unavailable. Reconstructed components include:

- workbook parsing and normalized event schema;
- time normalization and event-duration reconstruction;
- keyword-based activity categorization;
- participant-day feature engineering;
- HDBSCAN execution wrapper;
- TF-IDF and Sentence-BERT helper functions;
- routine-text similarity utilities;
- descriptive mood/category and feature/mood analysis;
- the CLI runner and structured outputs.

## ENHANCED

Modern portfolio-quality additions include:

- explicit configuration validation;
- defensive workbook and clock-value parsing;
- non-negative-duration guardrails;
- explicit separation of mood from clustering inputs;
- insufficient-sample handling for HDBSCAN;
- cluster diagnostics and assignment-strength reporting;
- data-quality diagnostics;
- deterministic t-SNE configuration;
- injected embedding-model support for offline tests;
- input/output contracts and implementation-status documentation;
- automated tests;
- Python 3.11–3.13 CI;
- Ruff linting;
- dependency-security auditing;
- full runtime import-compatibility validation.

## UNVERIFIED / NOT PRESENTED AS HISTORICAL FACT

Without the authentic source workbook and original result artifacts, this repository does **not** claim as verified:

- an exact historical row/event count;
- exact historical HDBSCAN hyperparameters;
- an exact historical final cluster count;
- exact historical mood-correlation values;
- exact historical t-SNE coordinates;
- exact historical cluster-quality metrics;
- that the reconstructed public modules are byte-for-byte copies of the original code.

The authentic student diary workbook is intentionally not published in this repository.
