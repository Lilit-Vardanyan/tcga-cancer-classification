
import json
import joblib
import pandas as pd
import streamlit as st

# -----------------------------
# 1. Load the saved model
# -----------------------------
model = joblib.load("random_forest_cancer_model.joblib")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

# -----------------------------
# 2. Configure the dashboard
# -----------------------------
st.set_page_config(
    page_title="Cancer Type Classification",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Cancer Type Classification")
st.write(
    "Upload a CSV file containing gene-expression features "
    "to explore the predictions made by the trained ML model."
)

st.info(
    "This is an educational machine-learning prototype, "
    "not a medical diagnostic tool."
)

st.write(f"**Expected gene-expression features:** {len(feature_names):,}")
st.write("**Model:** Tuned Random Forest Classifier")

# -----------------------------
# 3. Upload a CSV file
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload gene-expression data (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        # Compare column names as strings because CSV
        # column names are usually read as strings.
        data.columns = data.columns.astype(str)
        expected_names = [str(col) for col in feature_names]

        # Detect duplicate column names
        if data.columns.duplicated().any():
            st.error("Your CSV contains duplicate column names.")

        else:
            # Check whether all required genes are present
            missing = [
                col for col in expected_names
                if col not in data.columns
            ]

            if missing:
                st.error(
                    f"Your CSV is missing {len(missing)} required "
                    "gene-expression features."
                )
                st.write("Examples of missing features:", missing[:10])

            else:
                # Select the required columns in training order.
                # Extra columns, such as a 'cancer' label, are ignored.
                X_input = data.loc[:, expected_names].copy()

                # Restore the original training column names/types.
                X_input.columns = feature_names

                # Ensure every gene-expression value is numeric
                X_input = X_input.apply(
                    pd.to_numeric, errors="coerce"
                )

                if X_input.isna().any().any():
                    st.error(
                        "Some gene-expression values are missing "
                        "or non-numeric. Please correct the CSV."
                    )

                else:
                    st.success(
                        f"Successfully loaded {len(X_input)} sample(s)."
                    )

                    st.subheader("Uploaded data")
                    st.dataframe(data.head())

                    # -----------------------------
                    # 4. Make predictions
                    # -----------------------------
                    if st.button("Predict cancer type"):
                        predictions = model.predict(X_input)

                        results = pd.DataFrame({
                            "Sample": range(1, len(predictions) + 1),
                            "Predicted cancer type": predictions
                        })

                        st.subheader("Prediction results")
                        st.dataframe(results, use_container_width=True)

                        # Show class probability scores if available
                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(X_input)

                            st.subheader("Prediction scores")

                            for i, row in enumerate(probabilities):
                                st.write(f"**Sample {i + 1}**")

                                scores = pd.Series(
                                    row,
                                    index=model.classes_,
                                    name="Score"
                                ).sort_values(ascending=False)

                                st.bar_chart(scores)

                        # Offer results as a downloadable CSV
                        csv_results = results.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "Download predictions as CSV",
                            data=csv_results,
                            file_name="cancer_predictions.csv",
                            mime="text/csv"
                        )

    except Exception as e:
        st.error(f"Could not process the file: {e}")
