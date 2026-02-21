# Hansard Pharma-Relevance Classifier

A Streamlit labelling app for classifying parliamentary contributions by pharmaceutical relevance. Pulls contributions from a local Neo4j database of Hansard records, presents them for labelling, and stores results in SQLite to build a training dataset.

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

The app has three pages:

- **Ingest Data** — Pull contributions from Neo4j into SQLite. Includes an editable keyword list; removing a keyword purges its unlabelled contributions while preserving any already-labelled data.
- **Label Contributions** — Review contributions one at a time and label as Relevant / Not Relevant / Skip. Positive (keyword-matched) and negative (random) candidates are interleaved.
- **Review & Export** — View labelling progress and export the labelled dataset as CSV.

## Project Structure

| File | Description |
|---|---|
| `app.py` | Streamlit UI (3 pages) |
| `ingest.py` | Neo4j queries and keyword matching |
| `db.py` | SQLite schema, CRUD, and queue logic |
| `labelling.db` | SQLite database (created at runtime, gitignored) |
