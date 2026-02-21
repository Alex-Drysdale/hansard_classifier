"""Streamlit labelling app for parliamentary contribution classification."""

import json

import pandas as pd
import streamlit as st

import db
from ingest import ingest, DEFAULT_KEYWORDS

st.set_page_config(page_title="Pharma Relevance Labeller", layout="wide")

# Initialise keyword list in session state
if "keywords" not in st.session_state:
    st.session_state.keywords = list(DEFAULT_KEYWORDS)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
page = st.sidebar.radio("Navigation", ["Ingest Data", "Label Contributions", "Review & Export"])

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Ingest Data
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Ingest Data":
    st.header("Ingest Data from Neo4j")
    st.write("Pull keyword-matching and random contributions from Neo4j into the local SQLite database.")

    # ── Keyword editor ─────────────────────────────────────────────────
    st.subheader("Keywords")
    st.caption("Edit the list below (one keyword per line). Removing a keyword will purge its "
               "unlabelled contributions from the queue — already-labelled contributions are kept.")

    edited_text = st.text_area(
        "Keyword list",
        value="\n".join(st.session_state.keywords),
        height=200,
        label_visibility="collapsed",
    )
    # Parse back to list, stripping blanks
    edited_keywords = [kw.strip() for kw in edited_text.splitlines() if kw.strip()]

    # Show diff if keywords changed
    added = set(edited_keywords) - set(st.session_state.keywords)
    removed = set(st.session_state.keywords) - set(edited_keywords)
    if added or removed:
        cols = st.columns(2)
        if removed:
            cols[0].warning("Will remove: " + ", ".join(sorted(removed)))
        if added:
            cols[1].info("Will add: " + ", ".join(sorted(added)))

    st.divider()

    if st.button("Run Ingestion", type="primary"):
        # Commit keyword edits
        st.session_state.keywords = edited_keywords

        with st.spinner("Querying Neo4j…"):
            try:
                result = ingest(keywords=edited_keywords)
            except ConnectionError as e:
                st.error(str(e))
                st.stop()

        # Force labelling queue refresh on next visit
        st.session_state.refresh_queue = True

        st.success("Ingestion complete!")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positive candidates fetched", result["positives_fetched"])
        c2.metric("Negative candidates fetched", result["negatives_fetched"])
        c3.metric("Stale contributions purged", result["purged"])
        c4.metric("Total in DB", result["db_counts"]["total"])

    # Always show current DB state
    counts = db.get_counts()
    if counts["total"] > 0:
        st.divider()
        st.subheader("Current database")
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive candidates", counts["positives"])
        c2.metric("Negative candidates", counts["negatives"])
        c3.metric("Labelled so far", counts["labelled"])

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Label Contributions
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Label Contributions":
    st.header("Label Contributions")

    # Build / refresh queue
    if "queue" not in st.session_state or st.session_state.get("refresh_queue"):
        st.session_state.queue = db.get_unlabelled_queue()
        st.session_state.queue_idx = 0
        st.session_state.refresh_queue = False

    queue = st.session_state.queue
    idx = st.session_state.queue_idx

    # Progress
    counts = db.get_counts()
    total = counts["total"]
    labelled = counts["labelled"]

    if total == 0:
        st.info("No contributions in the database yet. Go to **Ingest Data** first.")
        st.stop()

    st.progress(labelled / total if total else 0)
    st.caption(f"{labelled} labelled out of {total} total")

    # Find next unlabelled from current position
    while idx < len(queue):
        # Verify it hasn't been labelled in the meantime
        conn = db.get_conn()
        already = conn.execute(
            "SELECT 1 FROM labels WHERE contribution_id = ?", (queue[idx]["id"],)
        ).fetchone()
        conn.close()
        if not already:
            break
        idx += 1

    if idx >= len(queue):
        st.success("All contributions in the current queue have been labelled!")
        if st.button("Refresh queue"):
            st.session_state.refresh_queue = True
            st.rerun()
        st.stop()

    item = queue[idx]
    matched = json.loads(item["matched_keywords"]) if item["matched_keywords"] else []

    # ── Contribution card ──────────────────────────────────────────────────
    with st.container(border=True):
        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.markdown(f"**{item['speaker']}** · {item['house']}")
            st.caption(f"{item['debate_title']}  —  {item['debate_date']}")
        with col_right:
            badge = "Keyword match" if item["is_candidate"] else "Random sample"
            colour = "green" if item["is_candidate"] else "gray"
            st.markdown(f":{colour}[{badge}]")

        st.markdown("---")
        st.markdown(item["text"])

        if matched:
            st.markdown("---")
            st.caption("Matched keywords: " + ", ".join(f"**{kw}**" for kw in matched))

    # ── Notes & buttons ────────────────────────────────────────────────────
    notes = st.text_input("Notes (optional)", key=f"notes_{item['id']}")

    col1, col2, col3 = st.columns(3)

    def _submit(label_val: int | None):
        if label_val is not None:
            db.save_label(item["id"], label_val, notes)
        st.session_state.queue_idx = idx + 1

    with col1:
        if st.button("Relevant", type="primary", use_container_width=True):
            _submit(1)
            st.rerun()
    with col2:
        if st.button("Not Relevant", type="secondary", use_container_width=True):
            _submit(0)
            st.rerun()
    with col3:
        if st.button("Skip", use_container_width=True):
            _submit(None)
            st.rerun()

    # Keyboard-friendly: show position
    remaining = len(queue) - idx
    st.caption(f"Position {idx + 1} of {len(queue)} in queue · ~{remaining} remaining")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Review & Export
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Review & Export":
    st.header("Review & Export")

    counts = db.get_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Relevant", counts["relevant"])
    c2.metric("Not Relevant", counts["not_relevant"])
    c3.metric("Total labelled", counts["labelled"])

    # Recent labels table
    st.subheader("Recent labels")
    recent = db.get_recent_labels(100)
    if recent:
        df = pd.DataFrame(recent)
        df["label"] = df["label"].map({1: "Relevant", 0: "Not Relevant"})
        st.dataframe(
            df[["contribution_id", "speaker", "debate_title", "debate_date", "label", "notes", "labelled_at"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No labels recorded yet.")

    # Export
    st.subheader("Export")
    export_rows = db.export_labelled()
    if export_rows:
        csv_df = pd.DataFrame(export_rows)
        csv_data = csv_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download labelled dataset (CSV)",
            data=csv_data,
            file_name="labelled_contributions.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption(f"{len(export_rows)} rows will be exported.")
    else:
        st.info("Nothing to export yet — label some contributions first.")
