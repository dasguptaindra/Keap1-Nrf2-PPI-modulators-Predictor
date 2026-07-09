# Keap1-Nrf2 pIC50 Predictor

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
```

## Important Descriptor Note

The model was trained on selected PaDEL fingerprints and RDKit topological torsion fingerprint bits. For SMILES prediction, the app attempts to calculate:

- `FP_*` columns with RDKit hashed topological torsion bits.
- `APC2D*` and `PubchemFP*` columns with PaDEL.
