# Student Routines & Mood Clustering

Behavioral analytics and unsupervised-learning project for transforming semi-structured student activity logs into event-level routine data and exploring relationships with self-reported mood.

## What it demonstrates

- Excel/multi-sheet data engineering
- Event reconstruction and duration inference
- TF-IDF
- Sentence-BERT (`all-MiniLM-L6-v2`)
- HDBSCAN clustering
- t-SNE visualization
- Pearson / Spearman analysis

## Pipeline

```text
Multi-sheet activity workbook
        |
        v
Long-format event reconstruction
        |
        v
Temporal + behavioral features
        |
        +--> TF-IDF
        +--> Sentence-BERT
        |
        v
HDBSCAN clustering
        |
        v
Cluster interpretation + mood analysis
```

The original workbook is not published in this repository. See `PROVENANCE.md` for recovered vs reconstructed details.
