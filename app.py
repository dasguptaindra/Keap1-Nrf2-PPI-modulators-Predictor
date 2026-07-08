from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from descriptor_utils import align_descriptor_frame, descriptors_from_smiles


ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "keap_gradient_boosting.joblib"

st.set_page_config(page_title="Keap1-NRF2 pIC50 Predictor", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"], .stMarkdown, .stDataFrame, .stButton button,
    .stDownloadButton button, .stTextInput input, .stTextArea textarea {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 22px !important;
        font-weight: 700 !important;
    }
    h1, h2, h3 {
        font-family: "Times New Roman", Times, serif !important;
        font-weight: 700 !important;
    }
    .metric-box {
        border: 2px solid #1f2937;
        padding: 14px;
        border-radius: 8px;
        background: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact():
    if not ARTIFACT_PATH.exists():
        st.error(
            "Model file is missing. Please upload "
            "`artifacts/keap_gradient_boosting.joblib` to GitHub, or upload "
            "`data/Train_keap.xlsx` and `data/Test_keap.xlsx` and run "
            "`python train_model.py` before deployment."
        )
        st.stop()
    return joblib.load(ARTIFACT_PATH)


artifact = load_artifact()
model = artifact["model"]
feature_names = artifact["feature_names"]
fingerprint_columns = artifact["fingerprint_columns"]
train_fingerprints = artifact["train_fingerprints"]
train_ids = artifact["train_compound_ids"]
ad_threshold = artifact["ad_threshold"]


def predict_frame(features):
    aligned = align_descriptor_frame(features, feature_names)
    preds = model.predict(aligned)
    ad_rows = [applicability_domain(aligned.iloc[[i]]) for i in range(len(aligned))]
    out = aligned.copy()
    out.insert(0, "Predicted_pIC50", preds)
    out.insert(1, "AD_Average_Top5_TS", [r["average_top5_ts"] for r in ad_rows])
    out.insert(2, "Within_AD", [r["within_ad"] for r in ad_rows])
    out.insert(3, "AD_Threshold", ad_threshold)
    return out, ad_rows


def applicability_domain(feature_row):
    query = feature_row.loc[:, fingerprint_columns].fillna(0).astype(int).to_numpy()[0]
    train = train_fingerprints.to_numpy().astype(int)
    sims = tanimoto_to_training(query, train)
    top_idx = np.argsort(sims)[::-1][:5]
    avg = float(np.mean(sims[top_idx])) if len(top_idx) else 0.0
    return {
        "average_top5_ts": avg,
        "within_ad": bool(avg > ad_threshold),
        "top_ids": [train_ids[i] for i in top_idx],
        "top_sims": [float(sims[i]) for i in top_idx],
    }


def tanimoto_to_training(query, train):
    query = query.astype(bool)
    train = train.astype(bool)
    inter = np.logical_and(train, query).sum(axis=1)
    union = np.logical_or(train, query).sum(axis=1)
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def shap_values(feature_row):
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(feature_row)
        if isinstance(values, list):
            values = values[0]
        base = float(np.ravel(explainer.expected_value)[0])
        return np.ravel(values), base
    except Exception:
        baseline = artifact["train_features"].mean(axis=0).to_frame().T
        pred = float(model.predict(feature_row)[0])
        base = float(model.predict(baseline)[0])
        diff = (feature_row.iloc[0] - baseline.iloc[0]).to_numpy(dtype=float)
        weights = getattr(model, "feature_importances_", np.ones(len(feature_names)))
        raw = diff * weights
        total = raw.sum()
        values = raw * ((pred - base) / total) if total != 0 else raw
        return values, base


def plot_waterfall(feature_row, prediction):
    values, base = shap_values(feature_row)
    order = np.argsort(np.abs(values))[::-1][:10]
    labels = [feature_names[i] for i in order]
    contrib = values[order]
    colors = ["#166534" if v >= 0 else "#b91c1c" for v in contrib]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels[::-1], contrib[::-1], color=colors[::-1])
    ax.axvline(0, color="#111827", linewidth=1.4)
    ax.set_title(f"Top 10 descriptor effects | Predicted pIC50 = {prediction:.3f}",
                 fontsize=24, fontweight="bold", fontname="Times New Roman")
    ax.set_xlabel("Contribution to prediction", fontsize=22, fontweight="bold", fontname="Times New Roman")
    ax.tick_params(axis="both", labelsize=18)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight("bold")
    fig.tight_layout()
    return fig


def plot_ad(ad):
    labels = ad["top_ids"]
    sims = ad["top_sims"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, sims, color="#2563eb")
    ax.axhline(ad_threshold, color="#b91c1c", linestyle="--", linewidth=2.5, label="AD threshold")
    ax.scatter(["Query"], [ad["average_top5_ts"]], marker="*", s=700, color="#f59e0b",
               edgecolor="#111827", linewidth=1.5, label="Query average top 5 TS")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Tanimoto similarity", fontsize=22, fontweight="bold", fontname="Times New Roman")
    ax.set_title("Applicability Domain", fontsize=24, fontweight="bold", fontname="Times New Roman")
    ax.legend(prop={"family": "Times New Roman", "weight": "bold", "size": 16})
    ax.tick_params(axis="both", labelsize=17)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight("bold")
    fig.tight_layout()
    return fig


def read_uploaded_table(file):
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


st.title("Keap1-NRF2 pIC50 Predictor")
tab_single, tab_batch = st.tabs(["Single compound", "Batch upload"])

with tab_single:
    st.subheader("Sketch or enter SMILES")
    smiles = ""
    try:
        from streamlit_ketcher import st_ketcher

        drawn = st_ketcher(height=450)
        if drawn:
            smiles = drawn
    except Exception:
        st.info("Chemical sketcher is unavailable. Enter SMILES below.")

    smiles = st.text_input("SMILES", value=smiles, placeholder="Example: CC(=O)Oc1ccccc1C(=O)O")
    descriptor_file = st.file_uploader(
        "Optional: upload one-row CSV/XLSX with model descriptors", type=["csv", "xlsx"], key="single_desc"
    )

    if st.button("Predict single compound"):
        try:
            if descriptor_file is not None:
                features = align_descriptor_frame(read_uploaded_table(descriptor_file), feature_names).iloc[[0]]
            elif smiles.strip():
                features = descriptors_from_smiles(smiles.strip(), feature_names)
            else:
                st.error("Please sketch a molecule, enter SMILES, or upload descriptors.")
                st.stop()

            result, ad_rows = predict_frame(features)
            pred = float(result["Predicted_pIC50"].iloc[0])
            ad = ad_rows[0]
            st.markdown(
                f"<div class='metric-box'>Predicted activity: {pred:.3f} pIC50<br>"
                f"Applicability domain: {'Within AD' if ad['within_ad'] else 'Outside AD'} "
                f"(Average top 5 TS = {ad['average_top5_ts']:.3f}; threshold = {ad_threshold:.3f})</div>",
                unsafe_allow_html=True,
            )
            left, right = st.columns(2)
            with left:
                st.pyplot(plot_waterfall(features, pred))
            with right:
                st.pyplot(plot_ad(ad))
        except Exception as exc:
            st.error(str(exc))

with tab_batch:
    st.subheader("Upload CSV or Excel")
    batch_file = st.file_uploader("File must contain either SMILES or the 51 descriptor columns", type=["csv", "xlsx"])
    if batch_file is not None:
        try:
            batch = read_uploaded_table(batch_file)
            if "SMILES" in batch.columns:
                rows = []
                errors = []
                for i, smi in enumerate(batch["SMILES"].astype(str)):
                    try:
                        rows.append(descriptors_from_smiles(smi, feature_names))
                        errors.append("")
                    except Exception as exc:
                        rows.append(pd.DataFrame([{c: 0 for c in feature_names}]))
                        errors.append(str(exc))
                features = pd.concat(rows, ignore_index=True)
            else:
                features = align_descriptor_frame(batch, feature_names)
                errors = [""] * len(features)

            predictions, _ = predict_frame(features)
            summary = batch.copy()
            summary["Predicted_pIC50"] = predictions["Predicted_pIC50"].values
            summary["AD_Average_Top5_TS"] = predictions["AD_Average_Top5_TS"].values
            summary["Within_AD"] = predictions["Within_AD"].map({True: "Yes", False: "No"}).values
            summary["AD_Threshold"] = ad_threshold
            if any(errors):
                summary["Descriptor_Error"] = errors
            st.dataframe(summary, use_container_width=True)
            st.download_button(
                "Download prediction results",
                summary.to_csv(index=False).encode("utf-8"),
                file_name="keap_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(str(exc))
