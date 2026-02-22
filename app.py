"""Streamlit labelling app for parliamentary contribution classification."""

import json

import pandas as pd
import streamlit as st

import db
import model
from ingest import ingest, DEFAULT_KEYWORDS

st.set_page_config(page_title="Pharma Relevance Labeller", layout="wide")

# Initialise keyword list in session state (prefer DB, fall back to defaults)
if "keywords" not in st.session_state:
    saved = db.get_keywords()
    st.session_state.keywords = saved if saved is not None else list(DEFAULT_KEYWORDS)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Ingest Data", "Label Contributions", "Review & Export", "Model Performance"],
)

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
        # Commit keyword edits to session state and database
        st.session_state.keywords = edited_keywords
        db.save_keywords(edited_keywords)

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
# PAGE 2 — Label Contributions (with active learning)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Label Contributions":

    # ── Sidebar: model settings ────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.subheader("Model Settings")
        retrain_interval = st.slider(
            "Retrain every N labels", 10, 100, 25, key="retrain_interval"
        )
        uncertainty_sampling = st.toggle("Uncertainty sampling", key="uncertainty_sampling")

        model_info = model.get_latest_model_info()
        if model_info:
            st.caption(f"Model v{model_info['id']} — {model_info['trained_at'][:16]}")
        else:
            st.caption("No model trained yet")

    st.header("Label Contributions")

    # ── Initialise model in session state ──────────────────────────────────
    if "model_pipeline" not in st.session_state:
        st.session_state.model_pipeline = model.load_latest_model()

    if "labels_since_train" not in st.session_state:
        st.session_state.labels_since_train = 0

    # Detect uncertainty-sampling toggle change → refresh queue
    if "prev_uncertainty" not in st.session_state:
        st.session_state.prev_uncertainty = uncertainty_sampling
    elif st.session_state.prev_uncertainty != uncertainty_sampling:
        st.session_state.refresh_queue = True
        st.session_state.prev_uncertainty = uncertainty_sampling

    # ── Build / refresh queue ──────────────────────────────────────────────
    if "queue" not in st.session_state or st.session_state.get("refresh_queue"):
        if uncertainty_sampling and st.session_state.model_pipeline is not None:
            with st.spinner("Scoring queue by uncertainty…"):
                st.session_state.queue = model.get_uncertainty_queue(
                    st.session_state.model_pipeline
                )
        else:
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

    # ── FEEDBACK MODE ──────────────────────────────────────────────────────
    if st.session_state.get("show_feedback"):
        feedback = st.session_state.last_feedback

        with st.container(border=True):
            if feedback["model_agreed"]:
                st.success("### ✓ Model agreed")
            else:
                pred_label = "Relevant" if feedback["model_prediction"] == 1 else "Not Relevant"
                # Show confidence in the model's own prediction
                if feedback["model_prediction"] == 1:
                    conf_pct = feedback["model_confidence"] * 100
                else:
                    conf_pct = (1 - feedback["model_confidence"]) * 100
                st.error(
                    f"### ✗ Model disagreed\n"
                    f"Predicted **{pred_label}** with {conf_pct:.0f}% confidence"
                )

        if st.button("Next", type="primary", use_container_width=True):
            st.session_state.show_feedback = False
            st.session_state.queue_idx = idx + 1

            # Check retrain trigger
            if (st.session_state.labels_since_train >= retrain_interval
                    and counts["labelled"] >= model.MIN_TRAINING_SAMPLES):
                with st.spinner("Retraining model…"):
                    try:
                        metrics = model.train_model()
                        st.session_state.model_pipeline = model.load_latest_model()
                        st.session_state.labels_since_train = 0
                        st.session_state.refresh_queue = True
                        st.toast(f"Model retrained! Accuracy: {metrics['accuracy']:.1%}")
                    except Exception as e:
                        st.toast(f"Retrain failed: {e}", icon="⚠️")

            st.rerun()

        st.stop()

    # ── LABELLING MODE ─────────────────────────────────────────────────────
    # Find next unlabelled from current position
    while idx < len(queue):
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

    st.session_state.queue_idx = idx
    item = queue[idx]

    # Get model prediction silently (don't show to user yet)
    current_prediction = None
    if st.session_state.model_pipeline is not None:
        pred, conf = model.predict_contribution(st.session_state.model_pipeline, item)
        current_prediction = {"prediction": pred, "confidence": conf}

    # ── Contribution card ──────────────────────────────────────────────────
    matched = json.loads(item["matched_keywords"]) if item["matched_keywords"] else []

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

    def _submit(label_val: int | None):
        if label_val is not None:
            model_pred = current_prediction["prediction"] if current_prediction else None
            model_conf = current_prediction["confidence"] if current_prediction else None
            model_agreed = None
            if model_pred is not None:
                model_agreed = 1 if model_pred == label_val else 0

            db.save_label(
                item["id"], label_val, notes,
                model_prediction=model_pred,
                model_confidence=model_conf,
                model_agreed=model_agreed,
            )
            st.session_state.labels_since_train += 1

            # If model is loaded, show feedback before advancing
            if current_prediction is not None:
                st.session_state.show_feedback = True
                st.session_state.last_feedback = {
                    "model_prediction": model_pred,
                    "model_confidence": model_conf,
                    "model_agreed": model_agreed,
                    "user_label": label_val,
                }
            else:
                # No model → advance immediately
                st.session_state.queue_idx = idx + 1
        else:
            # Skip → advance without saving
            st.session_state.queue_idx = idx + 1

    col1, col2, col3 = st.columns(3)

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

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.header("Model Performance")

    model_runs = db.get_model_runs()

    if not model_runs:
        st.info("No models have been trained yet. Labels with a model will appear "
                "once the first model is trained (requires 50+ labelled samples).")
        st.stop()

    # ── Current model metrics ──────────────────────────────────────────────
    latest = model_runs[-1]
    st.subheader("Current Model")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Version", f"v{latest['id']}")
    c2.metric("Accuracy", f"{latest['accuracy']:.3f}")
    c3.metric("Precision", f"{latest['precision_score']:.3f}")
    c4.metric("Recall", f"{latest['recall_score']:.3f}")
    c5.metric("F1", f"{latest['f1_score']:.3f}")
    st.caption(f"Trained on {latest['num_training_samples']} samples · {latest['trained_at'][:16]}")

    # ── Rolling agreement rate ─────────────────────────────────────────────
    st.divider()
    st.subheader("Model Convergence")
    agreement = db.get_rolling_agreement()
    if agreement is not None:
        target = 0.90
        st.progress(min(agreement, 1.0))
        st.caption(
            f"Rolling agreement rate (last 50 labels): **{agreement:.1%}** — target: {target:.0%}"
        )
        if agreement >= target:
            st.success("Model has reached convergence target!")
    else:
        st.caption("No model predictions recorded yet — agreement tracking will "
                   "begin after labelling with a trained model.")

    # ── Metrics over time ──────────────────────────────────────────────────
    if len(model_runs) > 1:
        st.divider()
        st.subheader("Metrics Over Model Versions")
        chart_df = pd.DataFrame(model_runs)
        chart_df = chart_df[["id", "accuracy", "precision_score", "recall_score", "f1_score"]]
        chart_df = chart_df.rename(columns={
            "id": "Version",
            "accuracy": "Accuracy",
            "precision_score": "Precision",
            "recall_score": "Recall",
            "f1_score": "F1",
        })
        chart_df = chart_df.set_index("Version")
        st.line_chart(chart_df)

    # ── Confusion matrix ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Confusion Matrix (latest model, held-out test set)")
    cm_json = latest.get("confusion_matrix")
    if cm_json:
        cm = json.loads(cm_json)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Not Relevant", "Actual: Relevant"],
            columns=["Pred: Not Relevant", "Pred: Relevant"],
        )
        st.dataframe(cm_df)

        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Positives", tp)
        c2.metric("False Positives", fp)
        c3.metric("True Negatives", tn)
        c4.metric("False Negatives", fn)
    else:
        st.caption("No confusion matrix available.")

    # ── Recent disagreements ───────────────────────────────────────────────
    st.divider()
    st.subheader("Recent Disagreements")
    disagreements = db.get_recent_disagreements()
    if disagreements:
        dis_df = pd.DataFrame(disagreements)
        dis_df["label"] = dis_df["label"].map({1: "Relevant", 0: "Not Relevant"})
        dis_df["model_prediction"] = dis_df["model_prediction"].map(
            {1: "Relevant", 0: "Not Relevant"}
        )
        dis_df["model_confidence"] = dis_df["model_confidence"].apply(
            lambda x: f"{x:.1%}" if x is not None else "N/A"
        )
        dis_df["text"] = dis_df["text"].str[:200] + "…"
        st.dataframe(
            dis_df[["text", "speaker", "label", "model_prediction",
                     "model_confidence", "labelled_at"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No disagreements recorded yet.")

    # ── Run model on full dataset ──────────────────────────────────────────
    st.divider()
    st.subheader("Run Model on Full Dataset")

    can_run = agreement is not None and agreement >= 0.85
    if not can_run:
        if agreement is not None:
            st.warning(
                f"Rolling agreement rate is {agreement:.1%}. "
                f"Must reach 85% before running on full dataset."
            )
        else:
            st.warning("No agreement data yet. Label more contributions with a trained model.")

    if st.button("Run model on full dataset", disabled=not can_run, type="primary"):
        pipeline = model.load_latest_model()
        if pipeline is None:
            st.error("No model found on disk.")
        else:
            progress_bar = st.progress(0.0)
            with st.spinner("Running predictions…"):
                predictions = model.predict_all_unlabelled(
                    pipeline,
                    progress_callback=lambda p: progress_bar.progress(min(p, 1.0)),
                )
            progress_bar.progress(1.0)

            if not predictions:
                st.info("No unlabelled contributions to predict.")
            else:
                db.save_full_predictions(predictions)
                st.session_state.full_predictions = predictions

    # Show results (persists across reruns)
    if "full_predictions" in st.session_state and st.session_state.full_predictions:
        predictions = st.session_state.full_predictions

        n_total = len(predictions)
        n_relevant = sum(1 for p in predictions if p["prediction"] == 1)
        n_not_relevant = n_total - n_relevant

        st.success(f"Model predictions complete for {n_total} unlabelled contributions")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Relevant", n_relevant)
        c2.metric("Predicted Not Relevant", n_not_relevant)
        c3.metric("% Predicted Relevant", f"{n_relevant / n_total:.1%}")

        # Breakdown by house
        summary = db.get_full_predictions_summary()
        if summary:
            summary_df = pd.DataFrame(summary)
            summary_df["prediction"] = summary_df["prediction"].map(
                {1: "Relevant", 0: "Not Relevant"}
            )
            pivot = summary_df.pivot_table(
                index="house", columns="prediction",
                values="count", fill_value=0, aggfunc="sum",
            )
            pivot.columns.name = None
            pivot.index.name = "House"
            if "Relevant" in pivot.columns:
                pivot["% Relevant"] = (
                    pivot["Relevant"] / (pivot.get("Relevant", 0) + pivot.get("Not Relevant", 0)) * 100
                ).round(1).astype(str) + "%"
            st.markdown("**Breakdown by House**")
            st.dataframe(pivot, use_container_width=True)

        # Top predicted-relevant contributions
        pred_df = pd.DataFrame(predictions)
        pred_df = pred_df.sort_values("confidence", ascending=False)
        top_relevant = pred_df[pred_df["prediction"] == 1].head(20)
        if not top_relevant.empty:
            # Fetch full details for the top predictions
            unlabelled = {r["id"]: r for r in db.get_unlabelled_contributions()}
            display_rows = []
            for _, row in top_relevant.iterrows():
                detail = unlabelled.get(row["id"], {})
                display_rows.append({
                    "Speaker": detail.get("speaker", ""),
                    "Debate": detail.get("debate_title", ""),
                    "Date": detail.get("debate_date", ""),
                    "Confidence": f"{row['confidence']:.1%}",
                    "Text": (detail.get("text", "")[:150] + "…")
                           if len(detail.get("text", "")) > 150 else detail.get("text", ""),
                })
            st.markdown("**Top 20 predicted-relevant contributions**")
            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

        # Download full predictions
        export_df = pred_df.copy()
        export_df["prediction"] = export_df["prediction"].map({1: "Relevant", 0: "Not Relevant"})
        export_df = export_df.rename(columns={
            "id": "contribution_id",
            "prediction": "predicted_label",
            "confidence": "relevance_probability",
        })
        csv_data = export_df[["contribution_id", "predicted_label", "relevance_probability", "house"]]\
            .to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download all predictions (CSV)",
            data=csv_data,
            file_name="model_predictions.csv",
            mime="text/csv",
        )
