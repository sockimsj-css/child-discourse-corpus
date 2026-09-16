# child-discourse-corpus

**Computational text analysis of how "the child" is discursively constructed across academic disciplines.**

> How do different academic fields rhetorically frame children — and does the choice of label (*child*, *youth*, *juvenile*, *adolescent*, *minor*) carry distinct connotations?

## Research Question

This project builds a large corpus of academic papers and uses computational text analysis (TF-IDF, co-occurrence networks, word embeddings) to map how age-related categories are linguistically represented across disciplines. The computational findings serve as a *discovery tool*; selected texts are then examined through qualitative close reading (CDA).

Theoretically grounded in the sociology of categorization (Zerubavel 1996; Johfre & Saperstein 2023; Barnes et al. 2024) and critical discourse analysis.

## Project Structure

```
child-discourse-corpus/
├── scripts/
│   ├── 01_collect_s2.py        # Semantic Scholar API collection
│   ├── 02_tfidf_quick.py       # TF-IDF analysis by term group
│   └── ...                     # (more scripts as project develops)
├── notebooks/                  # Jupyter notebooks for exploration
├── data/
│   ├── raw/                    # Raw API outputs (not committed)
│   └── processed/              # Cleaned datasets (not committed)
├── outputs/                    # Figures, tables, results
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Collect papers (Law field, pilot run)
python scripts/01_collect_s2.py --field Law --max-per-term 200

# Run quick TF-IDF analysis
python scripts/02_tfidf_quick.py
```

Optional: set `S2_API_KEY` environment variable for faster collection (~10x).

## Data Sources

- **Semantic Scholar API** (primary): titles, abstracts, metadata
- **OpenAlex API** (planned): broader coverage, cross-validation
- **CORE API** (planned): full-text access for selected papers

## Methods

| Stage | Method | Tool | Purpose |
|-------|--------|------|---------|
| 1. Discovery | TF-IDF | scikit-learn | Identify distinctive terms per label group |
| 2. Discovery | Co-occurrence network | networkx, VOSviewer | Map semantic neighborhoods |
| 3. Discovery | Word embeddings | gensim, sentence-transformers | Measure framing axes |
| 4. Justification | Close reading / CDA | Manual | Interpret and validate patterns |

## Status

🚧 **In progress** — Pilot phase (single-field collection and TF-IDF)

## References

- Nelson, L. K. (2020). Computational Grounded Theory. *Sociological Methods & Research*, 49(1), 3–42.
- Baker, P. et al. (2008). A useful methodological synergy? *Discourse & Society*, 19(3), 273–306.
- Barnes, L., Johfre, S., & Munsch, C. L. (2024). Galvanizing the "Missing Revolution." *AJS*, 130, 147–192.
- Johfre, S. & Saperstein, A. (2023). The Social Construction of Age. *Annual Review of Sociology*.

## License

MIT
