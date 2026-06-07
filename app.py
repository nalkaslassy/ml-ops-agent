import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import io
import joblib
import pandas as pd
import streamlit as st

from agent.router import route, route_and_get_class


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_compatible_models(df):
    """Return (model_name, reason) for every model that could work with this data."""
    compatible = []
    n = len(df)
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    text_cols    = [c for c in df.columns
                    if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]
    binary_01    = [c for c in numeric_cols
                    if df[c].nunique() == 2 and set(df[c].dropna().unique()).issubset({0, 1})]
    binary_any   = [c for c in df.columns if df[c].nunique() == 2]

    if n >= 200 and binary_01 and len([c for c in df.columns if c not in binary_01]) >= 2:
        compatible.append(("ChurnModel",  "has a binary (0/1) target column and multiple feature columns"))
        compatible.append(("FraudModel",  "has a binary (0/1) target column — works if the rare class represents fraud"))
    if n >= 200 and text_cols and binary_any:
        compatible.append(("SentimentModel", "has a text column and a binary label column"))
    if n >= 50 and len(numeric_cols) >= 2:
        compatible.append(("SegmentationModel", "has numeric feature columns that can be grouped without a label"))
    return compatible


ALGO_WHY = {
    "Logistic Regression": (
        "Logistic Regression won, which means the relationship between your features and the outcome "
        "is relatively linear. A simpler model generalised best on your data — this is actually a good "
        "sign, as simpler models tend to be more reliable and easier to interpret."
    ),
    "XGBoost": (
        "XGBoost won, which means your data contains complex non-linear patterns or interactions between "
        "features that tree-based ensemble methods are better at capturing. Gradient boosting iteratively "
        "corrects its own errors, making it powerful for structured tabular data."
    ),
    "Random Forest": (
        "Random Forest won by combining hundreds of decision trees and averaging their results. This "
        "typically happens when individual features interact in non-obvious ways and the data benefits "
        "from an ensemble approach to reduce overfitting."
    ),
    "Linear SVC": (
        "Linear SVC won, which is the most common outcome for text classification with TF-IDF features. "
        "It finds an optimal linear boundary between classes in high-dimensional sparse feature space, "
        "where most other algorithms struggle."
    ),
    "Naive Bayes": (
        "Naive Bayes won, suggesting your text features are relatively independent of each other and "
        "the classes are well-separated. It is extremely fast and works particularly well when the "
        "dataset is small or the vocabulary is large."
    ),
}

SEGMENTATION_WHY = {
    "K-Means": (
        "K-Means found the clearest cluster separation at this number of groups. It works best when "
        "your customers form roughly spherical, similarly-sized clusters in feature space."
    ),
    "DBSCAN": (
        "DBSCAN outperformed K-Means, meaning your customer groups are not uniformly sized or shaped. "
        "It finds clusters based on density rather than distance from a centre, which handles "
        "irregular shapes and automatically identifies outliers as noise."
    ),
}


def explain_winner(model_name, winner, winner_score, all_results):
    """Plain-English explanation of why the winning algorithm was chosen."""
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    second = sorted_results[1] if len(sorted_results) > 1 else None

    # Find matching explanation
    if model_name == "SegmentationModel":
        algo_type = "K-Means" if "K-Means" in winner else "DBSCAN"
        explanation = SEGMENTATION_WHY.get(algo_type, f"{winner} achieved the best silhouette score.")
    else:
        explanation = next(
            (exp for algo, exp in ALGO_WHY.items() if algo.lower() in winner.lower()),
            f"{winner} achieved the highest score on your specific dataset."
        )

    if second:
        margin = winner_score - second[1]
        if margin > 0.05:
            explanation += (
                f"\n\nIt clearly outperformed {second[0]} (gap: +{margin:.3f}), "
                f"which is a strong signal that this algorithm fits your data well."
            )
        elif margin > 0.01:
            explanation += (
                f"\n\nIt edged out {second[0]} by {margin:.3f}. Both are reasonable choices "
                f"for your data, but this one was consistently better across cross-validation folds."
            )
        else:
            explanation += (
                f"\n\nThe margin over {second[0]} is very small ({margin:.4f}) — both algorithms "
                f"performed nearly identically. The winner was selected by a slim margin."
            )
    return explanation


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="ML Ops Agent", layout="wide")
st.title("ML Ops Agent")
st.caption("Upload a dataset, describe what you want to do, and the agent picks and trains the best model automatically.")

tab_train, tab_predict, tab_about = st.tabs(["Train", "Predict", "About"])


# ── TRAIN ──────────────────────────────────────────────────────────────────────
with tab_train:

    # ── What we support + data format ─────────────────────────────────────────
    with st.expander("What models are supported and how should my data be formatted?", expanded=True):
        info_left, info_right = st.columns(2, gap="large")

        with info_left:
            st.markdown("**Supported models**")
            st.markdown(
                "| Model | What it does | Target column needed? |\n"
                "|---|---|---|\n"
                "| **Churn Prediction** | Predicts which customers will leave | Yes — binary (0 = stayed, 1 = left) |\n"
                "| **Fraud Detection** | Detects fraudulent transactions | Yes — binary (0 = legitimate, 1 = fraud) |\n"
                "| **Sentiment Analysis** | Classifies text as positive or negative | Yes — binary (0/1 or 'positive'/'negative') |\n"
                "| **Customer Segmentation** | Groups customers into natural segments | No |\n"
            )

        with info_right:
            st.markdown("**Data format requirements**")
            st.markdown(
                "- Upload a **CSV file** with column headers on the first row\n"
                "- Data should be **cleaned** — correct types, no obviously wrong values\n"
                "- **Missing values** are handled automatically\n"
                "- **Remove ID columns** (customer ID, transaction ID) using the exclude list — they add noise\n"
                "- **Minimum rows:** 200 for Churn / Fraud / Sentiment · 50 for Segmentation\n"
                "- For **Sentiment**, one column must contain the raw text (reviews, comments, etc.)\n"
                "- For **Churn / Fraud**, the target column must contain only `0` and `1` values, and both must be present"
            )

    st.divider()
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("1. Describe your problem")
        problem = st.text_input(
            "What do you want to do with this data?",
            placeholder="e.g. Which customers are likely to cancel their subscription?",
        )

        routed = None
        if problem:
            routed = route(problem, prefer_llm=False)
            st.success(f"**Model selected:** {routed['model']}")
            st.write(routed["reason"])
            with st.expander("Data requirements for this model"):
                for req in routed["data_needed"]:
                    st.write(f"• {req}")

        st.divider()

        st.subheader("2. Upload your dataset")
        st.caption("Upload a cleaned CSV file. Column headers must be on the first row.")
        uploaded = st.file_uploader("Drag and drop a CSV file", type=["csv"])

        df = None
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write(f"**{len(df):,} rows × {len(df.columns)} columns**")
            st.dataframe(df.head(), use_container_width=True)

    with right:
        if problem and df is not None and routed:
            # Allow user to override the routed model (when they clicked "use X instead")
            override = st.session_state.get("override_model")
            model_name = override if override else routed["model"]

            if override:
                st.info(f"Using **{override}** based on your selection.")
                if st.button("Reset to original model", key="reset_override"):
                    del st.session_state["override_model"]
                    st.session_state.pop("trained", None)
                    st.rerun()

            cols = df.columns.tolist()

            st.subheader("3. Configure columns")

            if model_name in ("ChurnModel", "FraudModel"):
                target = st.selectbox("Target column (what to predict)", cols)
                drop   = st.multiselect("Columns to exclude (IDs, keys)", [c for c in cols if c != target])
                init_kwargs = {"target_col": target, "drop_cols": drop}

            elif model_name == "SentimentModel":
                text_col  = st.selectbox("Text column", cols)
                label_col = st.selectbox("Label column", [c for c in cols if c != text_col])
                init_kwargs = {"text_col": text_col, "target_col": label_col}

            else:  # SegmentationModel
                drop = st.multiselect("Columns to exclude (IDs, keys)", cols)
                init_kwargs = {"drop_cols": drop}

            if model_name != "SegmentationModel":
                with st.expander("Advanced settings"):
                    n_iter = st.slider("Hyperparameter configs to test per algorithm", 1, 30, 10)
                    cv     = st.slider("Cross-validation folds", 2, 10, 5)
                    init_kwargs["n_iter"] = n_iter
                    init_kwargs["cv"]     = cv

            # ── Compatibility check ────────────────────────────────────────
            st.divider()
            st.subheader("4. Compatibility check")

            compatible_with_chosen = False
            try:
                ModelClass = route_and_get_class(problem, prefer_llm=False)["model_class"]
                if override:
                    # Load the override class directly
                    from agent.router import get_model_class
                    ModelClass = get_model_class(override)
                validate_kwargs = {k: v for k, v in init_kwargs.items() if k not in ("n_iter", "cv")}
                ModelClass(**validate_kwargs)._validate(df)
                st.success(f"Your data is compatible with **{model_name}**. Ready to train.")
                compatible_with_chosen = True
            except ValueError as e:
                st.error(f"**Not compatible with {model_name}:** {e}")

                # Only suggest alternatives — don't show the expander content here
                alternatives = list(dict.fromkeys(
                    m for m, _ in get_compatible_models(df) if m != model_name
                ))
                if alternatives:
                    st.warning(
                        f"Your data is not compatible with what you asked for, but it looks compatible "
                        f"with the following supported model(s): **{', '.join(alternatives)}**"
                    )
                    st.write("Would you like to use one of these instead?")
                    btn_cols = st.columns(len(alternatives))
                    for i, alt in enumerate(alternatives):
                        if btn_cols[i].button(f"Use {alt}", key=f"use_{alt}"):
                            st.session_state["override_model"] = alt
                            st.session_state.pop("trained", None)
                            st.rerun()
                else:
                    st.warning(
                        "Your data does not appear compatible with any of our supported models. "
                        "See the **About** tab for data requirements."
                    )

            # ── Train ──────────────────────────────────────────────────────
            st.divider()
            st.subheader("5. Train")

            if st.button(
                "Train Model",
                type="primary",
                use_container_width=True,
                disabled=not compatible_with_chosen,
            ):
                # Clear previous results
                st.session_state.pop("trained", None)
                with st.spinner(f"Training {model_name} — testing all algorithms, this may take a few minutes..."):
                    try:
                        if override:
                            from agent.router import get_model_class
                            ModelClass = get_model_class(override)
                        else:
                            ModelClass = route_and_get_class(problem, prefer_llm=False)["model_class"]
                        model = ModelClass(**init_kwargs)
                        model.fit(df)
                        st.session_state["model"]      = model
                        st.session_state["model_name"] = model_name
                        st.session_state["trained"]    = True
                    except Exception as e:
                        st.error(str(e))

            # ── Results ────────────────────────────────────────────────────
            if st.session_state.get("trained"):
                model = st.session_state["model"]
                mn    = st.session_state["model_name"]

                st.success("Training complete")

                col_a, col_b = st.columns(2)
                col_a.metric("Best algorithm", model.best_name)
                col_b.metric("CV score", f"{model.best_score:.4f}")

                # Algorithm comparison table
                if model.results_:
                    st.subheader("How every algorithm compared")
                    rows = sorted(model.results_.items(), key=lambda x: x[1], reverse=True)
                    table_data = {
                        "Algorithm": [r[0] for r in rows],
                        "Score":     [f"{r[1]:.4f}" for r in rows],
                        "Result":    ["Winner" if r[0] == model.best_name else "" for r in rows],
                    }
                    st.table(pd.DataFrame(table_data))

                # Plain-English explanation
                st.subheader("Why this algorithm was chosen")
                st.write(explain_winner(mn, model.best_name, model.best_score, model.results_))

                st.divider()

                buf = io.BytesIO()
                joblib.dump(model, buf)
                buf.seek(0)
                st.download_button(
                    "Download trained model (.pkl)",
                    data=buf,
                    file_name=f"{mn.lower()}.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
                st.caption(
                    "Save this file. Upload it in the **Predict** tab to run predictions "
                    "on new data at any time — no retraining needed."
                )


# ── PREDICT ────────────────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("Load a trained model and run predictions on new data")
    st.write("Upload the `.pkl` file you downloaded after training, plus a CSV with new rows to predict on.")

    left, right = st.columns(2, gap="large")

    with left:
        model_file = st.file_uploader("Trained model (.pkl)", type=["pkl"])
        data_file  = st.file_uploader("New data (CSV)", type=["csv"])

        if model_file and data_file:
            model  = joblib.load(model_file)
            new_df = pd.read_csv(data_file)
            st.write(f"**Model:** {type(model).__name__} — {model.best_name} (score: {model.best_score:.4f})")
            st.write(f"**Data:** {len(new_df):,} rows × {len(new_df.columns)} columns")
            st.dataframe(new_df.head(), use_container_width=True)

    with right:
        if model_file and data_file:
            if st.button("Run Predictions", type="primary", use_container_width=True):
                try:
                    if type(model).__name__ == "SentimentModel":
                        preds = model.predict(new_df[model.text_col])
                    else:
                        preds = model.predict(new_df)

                    st.success(f"{len(preds):,} predictions generated")
                    st.dataframe(preds, use_container_width=True)
                    st.download_button(
                        "Download predictions (CSV)",
                        data=preds.to_csv(index=False).encode(),
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(str(e))


# ── ABOUT ──────────────────────────────────────────────────────────────────────
with tab_about:
    st.header("What this does")
    st.write(
        "Describe your business problem in plain English, upload a cleaned CSV dataset, and the agent "
        "selects the right model type and finds the best algorithm for your specific data. "
        "If your data is not compatible with what you asked for, it will tell you why and show you "
        "which of the supported models your data does work with."
    )

    st.header("How to prepare your data")
    st.write(
        "- Upload a **CSV file** with column headers on the first row\n"
        "- The data should be **reasonably clean** — correct column types, no obviously wrong values\n"
        "- **Missing values** are handled automatically via imputation\n"
        "- **ID columns** (customer ID, transaction ID, etc.) should be excluded using the exclude list — "
        "they add noise and won't help the model\n"
        "- **Minimum rows:** 200 for Churn, Fraud and Sentiment; 50 for Segmentation"
    )

    st.header("Supported models")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Churn Prediction")
        st.write("Predicts which customers are likely to leave.")
        st.caption(
            "Requires: customer feature columns (numeric or categorical) + "
            "a binary target column where 0 = stayed and 1 = left."
        )

        st.subheader("Fraud Detection")
        st.write("Detects fraudulent transactions. Built specifically for imbalanced data where fraud is rare.")
        st.caption(
            "Requires: transaction feature columns + "
            "a binary target column where 0 = legitimate and 1 = fraud. "
            "Both classes must be present."
        )

    with col2:
        st.subheader("Sentiment Analysis")
        st.write("Classifies text (reviews, feedback, comments) as positive or negative.")
        st.caption(
            "Requires: a text column containing the raw text + "
            "a binary label column (0/1 or 'positive'/'negative')."
        )

        st.subheader("Customer Segmentation")
        st.write("Groups customers into natural segments. No label column needed.")
        st.caption(
            "Requires: numeric feature columns only. "
            "No target column — the model finds the groupings automatically."
        )

    st.header("How the algorithm is selected")
    st.write(
        "For each model the agent tests multiple algorithms across many hyperparameter configurations "
        "using cross-validation, picks whichever scores best on your data, and shows you how every "
        "algorithm compared and why the winner was chosen."
    )

    algo_data = {
        "Model":            ["Churn", "Fraud", "Sentiment", "Segmentation"],
        "Algorithms tested": [
            "XGBoost, Random Forest, Logistic Regression",
            "XGBoost, Random Forest, Logistic Regression (all imbalance-aware)",
            "Logistic Regression, Linear SVC, Naive Bayes, Random Forest, XGBoost",
            "K-Means (k=2–8), DBSCAN (multiple settings)",
        ],
        "Scored by": ["ROC-AUC", "Precision-Recall AUC", "ROC-AUC", "Silhouette score"],
    }
    st.table(pd.DataFrame(algo_data))

    st.header("Saving and reusing your model")
    st.write(
        "After training, download the `.pkl` file. It contains the fully trained model — "
        "the winning algorithm, all fitted parameters, and the preprocessing steps. "
        "Upload it in the **Predict** tab to run predictions on new data at any time without retraining."
    )
    st.code(
        "import joblib\nmodel = joblib.load('my_model.pkl')\npredictions = model.predict(new_df)",
        language="python",
    )
