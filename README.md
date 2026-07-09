# Keap1-Nrf2-PPI-modulators-Predictor
Welcome to the Keap1-Nrf2-PPI-modulators Predictor, a ML-based web application designed for researchers, medicinal chemists and drug developers

💡 Why Use This Tool?

✨ Easy to use - Input SMILES and get instant prediction

🧠 AI-Driven Results - Predict Keap1-Nrf2-PPI-modulators activity with a trained ML model

🧬 Explainable Predictions - SHAP Waterfall plot to interpret feature contributions

⚗️ Useful for Drug Discovery

🌐 Accessible Anywhere - Hosted online via Streamlit

👇 Click the link to start predicting! 🔗 Streamlit App

🔬 Background

This model supports virtual screening workflows and accelerates early-stage drug discovery.
## Files
- `app.py`
- `train_model.py`
- `descriptor_utils.py`
- `requirements.txt`
- `data/Train_keap.xlsx`
- `data/Test_keap.xlsx`
- `artifacts/`: generated automatically after training.


## Input Options

**Single compound:**

- Sketch a molecule with the Ketcher widget.
- Or enter SMILES.

**Batch mode:**

- Upload CSV/XLSX containing a `SMILES` column.
- The app returns predicted pIC50, average top-5 Tanimoto similarity, AD threshold, and whether each compound is within the applicability domain.

## Applicability Domain

For each query, the app compares its fingerprint bits against all training compounds and takes the average Tanimoto similarity of the top 5 most similar training compounds. The AD threshold is the median, or 50th percentile, of all pairwise training-set Tanimoto similarities. A compound is marked within AD when:

```text
average top 5 Tanimoto similarity > threshold
``
## Important Descriptor Note

The model was trained on selected PaDEL fingerprints and RDKit topological torsion fingerprint bits. For SMILES prediction, the app attempts to calculate:

- `FP_*` columns with RDKit hashed topological torsion bits.
- `APC2D*` and `PubchemFP*` columns with PaDEL.

⭐ Support

If you find this tool useful, share it with researchers and students!

