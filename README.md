# Hansard Pharma-Relevance Classifier

A Streamlit app for classifying parliamentary contributions by pharmaceutical relevance. Pulls contributions from a local Neo4j database of Hansard records, presents them for labelling with active-learning model feedback, and trains a TF-IDF + Logistic Regression classifier to prioritise and predict relevance across the full dataset.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Neo4j credentials:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

## Usage

```bash
streamlit run app.py
```

The app has four pages:

### Ingest Data
Pull contributions from Neo4j into SQLite. Includes an editable keyword list that persists across sessions; removing a keyword purges its unlabelled contributions while preserving any already-labelled data. Negative (random) samples are capped at 500 and refreshed each run.

### Label Contributions
Review contributions one at a time and label as Relevant / Not Relevant / Skip. Includes active-learning features:

- **Model feedback** — after each label, the model's prediction is revealed (agree/disagree) so you can see where the model struggles.
- **Automatic retraining** — the model retrains every N labels (configurable via sidebar slider, default 25). Requires at least 50 labelled samples.
- **Uncertainty sampling** — toggle in sidebar to sort the queue by model uncertainty (contributions closest to the 0.5 decision boundary first), focusing labelling effort where it helps the model most.

When no model has been trained yet, the page works as a standard labelling interface.

### Review & Export
View labelling progress, recent labels, and export the labelled dataset as CSV.

### Model Performance
Track classifier performance across training runs:

- **Current model metrics** — accuracy, precision, recall, F1 from the held-out test set (80/20 stratified split).
- **Convergence tracking** — rolling agreement rate between the model and labeller over the last 50 labels, with a 90% target.
- **Metrics over time** — line chart of accuracy/precision/recall/F1 across model versions.
- **Confusion matrix** — TP/FP/TN/FN from the latest model's test set.
- **Disagreements table** — recent cases where the model and labeller disagreed, useful for spotting patterns.
- **Run model on full dataset** — predict relevance for all unlabelled contributions (enabled once agreement reaches 85%). Shows a breakdown by house and a downloadable CSV of all predictions.

## Project Structure

| File | Description |
|---|---|
| `app.py` | Streamlit UI (4 pages) |
| `ingest.py` | Neo4j queries and keyword matching |
| `db.py` | SQLite schema, CRUD, queue logic, and model run tracking |
| `model.py` | TF-IDF + Logistic Regression pipeline, training, evaluation, and prediction |
| `models/` | Versioned model files (`model_v1.pkl`, `model_v2.pkl`, …) |
| `labelling.db` | SQLite database (created at runtime, gitignored) |
| `requirements.txt` | Python dependencies |

## Model Details

- **Pipeline**: `TfidfVectorizer` (10k features, unigrams + bigrams, sublinear TF) + 3 numeric features (`is_candidate`, `word_count`, `num_keywords`) via `ColumnTransformer`, fed into `LogisticRegression` with `class_weight='balanced'`.
- **Evaluation**: 80/20 stratified train/test split. Metrics are computed on the held-out test set.
- **Versioning**: Each training run saves a new `models/model_v{n}.pkl` and logs metrics to the `model_runs` table.
- **Minimum samples**: 50 labelled contributions required before the first model can be trained.
