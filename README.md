# Keap1-NRF2 pIC50 Streamlit Predictor

This folder is ready to upload to GitHub and connect to Streamlit Community Cloud.

## Files

- `app.py`: Streamlit predictor app.
- `train_model.py`: trains the Gradient Boosting model and saves artifacts.
- `descriptor_utils.py`: SMILES descriptor generation and descriptor alignment.
- `requirements.txt`: Python packages for Streamlit Cloud.
- `data/Train_keap.xlsx`: training set.
- `data/Test_keap.xlsx`: external test set.
- `artifacts/`: generated automatically after training.

## Verified Model Result


## Local Run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## GitHub and Streamlit Cloud Deployment

1. Create a new GitHub repository.
2. Upload all files and folders in this directory to the repository root.
3. Confirm these files are visible in GitHub:
   - `app.py`
   - `descriptor_utils.py`
   - `requirements.txt`
   - `packages.txt`
   - `artifacts/keap_gradient_boosting.joblib`
   - `artifacts/metrics.json`
   - `data/Train_keap.xlsx`
   - `data/Test_keap.xlsx`
4. Go to Streamlit Community Cloud.
5. Choose the repository.
6. Set the main file path to `app.py`.
7. Deploy.

If you see `FileNotFoundError`, it means GitHub is missing either the `artifacts`
folder or the `data` folder. The app needs at least
`artifacts/keap_gradient_boosting.joblib` to start.

## Input Options

Single compound:

- Sketch a molecule with the Ketcher widget.
- Or enter SMILES.
- Or upload a one-row CSV/XLSX containing the 51 model descriptor columns.

Batch mode:

- Upload CSV/XLSX containing a `SMILES` column.
- Or upload CSV/XLSX containing the 51 descriptor columns.
- The app returns predicted pIC50, average top-5 Tanimoto similarity, AD threshold, and whether each compound is within the applicability domain.

## Applicability Domain

For each query, the app compares its fingerprint bits against all training compounds and takes the average Tanimoto similarity of the top 5 most similar training compounds. The AD threshold is the median, or 50th percentile, of all pairwise training-set Tanimoto similarities. A compound is marked within AD when:

```text
average top 5 Tanimoto similarity > threshold
```

The current model uses `PubchemFP*` and `FP_*` columns for fingerprint Tanimoto similarity.

## Important Descriptor Note

The model was trained on selected PaDEL descriptors/fingerprints and RDKit topological torsion fingerprint bits. For SMILES prediction, the app attempts to calculate:

- `FP_*` columns with RDKit hashed topological torsion bits.
- `APC2D*` and `PubchemFP*` columns with PaDEL through `padelpy`.

PaDEL may require Java on the deployment machine. If Streamlit Cloud cannot calculate PaDEL descriptors reliably, upload files that already contain the trained descriptor columns, or use a deployment environment where Java and PaDEL are available.
