import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None  # suppress blocking calls inside model fit()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import io
import joblib
import pandas as pd
import streamlit as st

from agent.router import route, route_and_get_class

st.set_page_config(page_title="ML Ops Agent", layout="wide")
st.title("ML Ops Agent")
st.caption("Describe your problem, upload a dataset, and the agent picks and trains the best model automatically.")

tab_train, tab_predict, tab_about = st.tabs(["Train", "Predict", "About"])


# ── TRAIN ──────────────────────────────────────────────────────────────────────
with tab_train:
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
            with st.expander("Data requirements"):
                for req in routed["data_needed"]:
                    st.write(f"• {req}")

        st.divider()

        st.subheader("2. Upload your dataset")
        uploaded = st.file_uploader("Drag and drop a CSV file", type=["csv"])

        df = None
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write(f"**{len(df):,} rows × {len(df.columns)} columns**")
            st.dataframe(df.head(), use_container_width=True)

    with right:
        if problem and df is not None and routed:
            model_name = routed["model"]
            cols = df.columns.tolist()

            st.subheader("3. Configure columns")

            if model_name in ("ChurnModel", "FraudModel"):
                target = st.selectbox("Target column (what to predict)", cols)
                drop = st.multiselect("Columns to exclude (IDs, keys)", [c for c in cols if c != target])
                init_kwargs = {"target_col": target, "drop_cols": drop}

            elif model_name == "SentimentModel":
                text_col = st.selectbox("Text column", cols)
                label_col = st.selectbox("Label column", [c for c in cols if c != text_col])
                init_kwargs = {"text_col": text_col, "target_col": label_col}

            else:  # SegmentationModel
                drop = st.multiselect("Columns to exclude (IDs, keys)", cols)
                init_kwargs = {"drop_cols": drop}

            with st.expander("Advanced settings"):
                n_iter = st.slider("Hyperparameter configs to test per algorithm", 1, 30, 10)
                cv = st.slider("Cross-validation folds", 2, 10, 5)
                init_kwargs["n_iter"] = n_iter
                init_kwargs["cv"] = cv

            st.divider()
            st.subheader("4. Train")

            if st.button("Train Model", type="primary", use_container_width=True):
                with st.spinner(f"Training {model_name} — testing all algorithms, this may take a few minutes..."):
                    try:
                        ModelClass = route_and_get_class(problem, prefer_llm=False)["model_class"]
                        model = ModelClass(**init_kwargs)
                        model.fit(df)

                        figs = [plt.figure(i) for i in plt.get_fignums()]
                        plt.close("all")

                        st.session_state["model"]      = model
                        st.session_state["model_name"] = model_name
                        st.session_state["figs"]       = figs
                        st.session_state["trained"]    = True
                    except Exception as e:
                        st.error(str(e))

            if st.session_state.get("trained"):
                model = st.session_state["model"]

                st.success("Training complete")

                col_a, col_b = st.columns(2)
                col_a.metric("Best algorithm", model.best_name)
                col_b.metric("CV score", f"{model.best_score:.4f}")

                for fig in st.session_state.get("figs", []):
                    st.pyplot(fig, use_container_width=True)

                buf = io.BytesIO()
                joblib.dump(model, buf)
                buf.seek(0)
                st.download_button(
                    "Download trained model (.pkl)",
                    data=buf,
                    file_name=f"{st.session_state['model_name'].lower()}.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
                st.info("Save this file. Upload it in the **Predict** tab to run predictions on new data — no retraining needed.")


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
        "Describe your business problem in plain English, upload a CSV dataset, "
        "and the agent selects the right model type and finds the best algorithm for your specific data. "
        "No ML knowledge required."
    )

    st.header("Supported models")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Churn Prediction")
        st.write("Predicts which customers are likely to leave.")
        st.caption("Data needed: customer feature columns + a binary churn column (0 = stayed, 1 = left). Minimum 200 rows.")

        st.subheader("Fraud Detection")
        st.write("Detects fraudulent transactions. Built for imbalanced data where fraud is rare.")
        st.caption("Data needed: transaction feature columns + a binary fraud column (0 = legitimate, 1 = fraud). Minimum 200 rows.")

    with col2:
        st.subheader("Sentiment Analysis")
        st.write("Classifies text (reviews, feedback, comments) as positive or negative.")
        st.caption("Data needed: a text column + a label column (0/1 or 'positive'/'negative'). Minimum 200 rows.")

        st.subheader("Customer Segmentation")
        st.write("Groups customers into natural segments automatically. No label column needed.")
        st.caption("Data needed: numeric feature columns only. Minimum 50 rows.")

    st.header("How the algorithm is selected")
    st.write(
        "For each model type the agent tests multiple algorithms across many hyperparameter "
        "configurations using cross-validation, then picks whichever scores best on your specific data:"
    )
    data = {
        "Model": ["Churn", "Fraud", "Sentiment", "Segmentation"],
        "Algorithms tested": [
            "XGBoost, Random Forest, Logistic Regression",
            "XGBoost, Random Forest, Logistic Regression (all imbalance-aware)",
            "Logistic Regression, Linear SVC, Naive Bayes, Random Forest, XGBoost",
            "K-Means (k=2–8), DBSCAN (multiple settings)",
        ],
        "Scored by": ["ROC-AUC", "Precision-Recall AUC", "ROC-AUC", "Silhouette score"],
    }
    st.table(pd.DataFrame(data))

    st.header("Saving and reusing your model")
    st.write(
        "After training, download the `.pkl` file. This contains the fully trained model — "
        "which algorithm won, all fitted parameters, preprocessing steps. "
        "Upload it in the **Predict** tab whenever you want predictions on new data. "
        "You can also load it directly in Python:"
    )
    st.code(
        "import joblib\n"
        "model = joblib.load('my_model.pkl')\n"
        "predictions = model.predict(new_df)",
        language="python",
    )

    st.header("Data requirements")
    st.write(
        "- Models handle missing values automatically via imputation\n"
        "- Your data should be reasonably clean (correct column types, no obvious errors)\n"
        "- If your data is not compatible, the model will tell you exactly what is wrong\n"
        "- Drop ID columns and other non-feature columns using the exclude list during configuration"
    )
