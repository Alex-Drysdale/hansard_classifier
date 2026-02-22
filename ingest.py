"""Neo4j data ingestion for the labelling app."""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

import db

load_dotenv()

DEFAULT_KEYWORDS = [
    "pharmaceutical", "pharma", "medicines", "mounjaro", "tirzepatide",
    "GLP-1", "anti-obesity medicines", "obesity", "obesity economic burden",
    "weight management", "Eli Lilly", "donanemab", "lecanemab", "amyloid",
    "tau", "neurodegeneration", "disease-modifying treatment", "Alzheimer",
    "dementia", "PET scan", "CSF testing", "blood biomarker", "biomarker",
    "genomic", "genetic testing", "selpercatinib", "breast cancer",
    "lung cancer", "Verzenio", "abemaciclib", "Trulicity", "Humalog",
    "Basaglar", "insulin pricing", "retatrutide", "Wegovy", "Ozempic",
    "Novo Nordisk", "diabetes", "life sciences", "MHRA",
    "clinical trial", "drug pricing", "medicine access", "QALY",
    "value-based pricing", "horizon scanning", "weight loss", "fat jabs",
    "fat loss drugs",
]


def _get_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    return GraphDatabase.driver(uri, auth=(user, password))


def _find_keywords(text: str, keywords: list[str]) -> list[str]:
    """Return which keywords appear in the text (case-insensitive)."""
    lower = text.lower()
    return [kw for kw in keywords if kw.lower() in lower]


def pull_positive_candidates(driver, keywords: list[str]) -> list[dict]:
    """Fetch contributions whose text matches at least one keyword."""
    query = """
        UNWIND $keywords AS kw
        MATCH (p)-[:MADE_CONTRIBUTION]->(c:Contribution)-[:CONTRIBUTION_IN]->(d:Debate)
        WHERE (p:MP OR p:Lord)
          AND c.word_count >= 20
          AND toLower(c.text) CONTAINS toLower(kw)
        RETURN DISTINCT c.id AS id, c.text AS text, c.word_count AS word_count,
               p.name AS speaker, p.house AS house,
               d.title AS debate_title, toString(d.date) AS debate_date
    """
    with driver.session() as session:
        results = session.run(query, keywords=keywords).data()

    rows = []
    for r in results:
        matched = _find_keywords(r["text"], keywords)
        rows.append({**r, "matched_keywords": matched, "is_candidate": 1})
    return rows


def pull_negative_candidates(driver, positive_ids: set[str], limit: int = 500) -> list[dict]:
    """Fetch random contributions that are NOT in the positive set."""
    query = """
        MATCH (p)-[:MADE_CONTRIBUTION]->(c:Contribution)-[:CONTRIBUTION_IN]->(d:Debate)
        WHERE (p:MP OR p:Lord)
          AND c.word_count >= 20
          AND NOT c.id IN $pos_ids
        RETURN c.id AS id, c.text AS text, c.word_count AS word_count,
               p.name AS speaker, p.house AS house,
               d.title AS debate_title, toString(d.date) AS debate_date,
               rand() AS r
        ORDER BY r
        LIMIT $limit
    """
    with driver.session() as session:
        results = session.run(query, pos_ids=list(positive_ids), limit=limit).data()

    return [
        {k: v for k, v in r.items() if k != "r"} | {"matched_keywords": [], "is_candidate": 0}
        for r in results
    ]


def ingest(keywords: list[str] | None = None) -> dict:
    """Run full ingestion. Returns summary counts.

    If keywords differ from a previous run, unlabelled positive candidates
    that no longer match are purged — labelled contributions are always kept.
    """
    keywords = keywords or DEFAULT_KEYWORDS

    driver = _get_driver()
    try:
        driver.verify_connectivity()
    except Exception as e:
        raise ConnectionError(f"Cannot connect to Neo4j: {e}") from e

    positives = pull_positive_candidates(driver, keywords)
    positive_ids = {r["id"] for r in positives}

    # Purge unlabelled positives that no longer match the current keyword list
    purged = db.purge_stale_unlabelled(positive_ids)

    # Replace the unlabelled negative pool (don't accumulate across runs)
    purged_neg = db.purge_unlabelled_negatives()

    negatives = pull_negative_candidates(driver, positive_ids)
    driver.close()

    db.insert_contributions(positives)
    db.insert_contributions(negatives)

    return {
        "positives_fetched": len(positives),
        "negatives_fetched": len(negatives),
        "purged": purged,
        "db_counts": db.get_counts(),
    }
