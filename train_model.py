from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
TRAIN_FILE = DATA_DIR / "Train_keap.xlsx"
TEST_FILE = DATA_DIR / "Test_keap.xlsx"
SMILES_LOOKUP_FILE = DATA_DIR / "smiles_descriptor_lookup.csv"

MODEL_PARAMS = {
    "n_estimators": 180,
    "max_depth": 14,
    "min_samples_split": 5,
    "min_samples_leaf": 9,
    "learning_rate": 0.027135056136594424,
    "subsample": 0.8074583548164399,
    "random_state": 42,
}


def load_xy(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}. Make sure the GitHub repository "
            "contains the data folder with Train_keap.xlsx and Test_keap.xlsx."
        )
    df = pd.read_excel(path)
    feature_names = list(df.columns[1:-1])
    x = df.loc[:, feature_names]
    y = df.iloc[:, -1].astype(float)
    return df, x, y, feature_names


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    missing = [p for p in (TRAIN_FILE, TEST_FILE) if not p.exists()]
    if missing:
        missing_text = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Cannot train the model because these required files are missing:\n"
            f"{missing_text}\n\n"
            "Upload the complete data folder to GitHub, or upload the prebuilt "
            "artifacts/keap_gradient_boosting.joblib file."
        )

    train_df, x_train, y_train, feature_names = load_xy(TRAIN_FILE)
    test_df, x_test, y_test, _ = load_xy(TEST_FILE)

    model = GradientBoostingRegressor(**MODEL_PARAMS)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    cv_result = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring)

    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    fingerprint_columns = [
        c for c in feature_names if c.startswith("PubchemFP") or c.startswith("FP_")
    ]
    fp_train = train_df.loc[:, fingerprint_columns].fillna(0).astype(int)
    threshold = pairwise_tanimoto_median(fp_train.to_numpy())
    smiles_lookup = load_smiles_lookup(feature_names)

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "fingerprint_columns": fingerprint_columns,
        "train_features": x_train,
        "train_fingerprints": fp_train,
        "train_compound_ids": train_df.iloc[:, 0].astype(str).tolist(),
        "ad_threshold": threshold,
        "baseline_prediction": float(np.mean(pred_train)),
        "target_name": train_df.columns[-1],
        "compound_id_name": train_df.columns[0],
        "model_params": MODEL_PARAMS,
        "smiles_lookup": smiles_lookup,
    }
    joblib.dump(artifact, ARTIFACT_DIR / "keap_gradient_boosting.joblib")

    metrics = {
        "cv": {
            "r2_mean": float(cv_result["test_r2"].mean()),
            "r2_std": float(cv_result["test_r2"].std()),
            "mae_mean": float((-cv_result["test_mae"]).mean()),
            "mae_std": float((-cv_result["test_mae"]).std()),
            "rmse_mean": float((-cv_result["test_rmse"]).mean()),
            "rmse_std": float((-cv_result["test_rmse"]).std()),
        },
        "train_fit": {
            "r2": float(r2_score(y_train, pred_train)),
            "mae": float(mean_absolute_error(y_train, pred_train)),
            "rmse": rmse(y_train, pred_train),
        },
        "external_test": {
            "r2": float(r2_score(y_test, pred_test)),
            "mae": float(mean_absolute_error(y_test, pred_test)),
            "rmse": rmse(y_test, pred_test),
        },
        "ad_threshold_median_pairwise_tanimoto": threshold,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(feature_names)),
        "n_smiles_lookup": int(len(smiles_lookup)),
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def pairwise_tanimoto_median(bits):
    bits = np.asarray(bits).astype(bool)
    sims = []
    for i in range(bits.shape[0] - 1):
        a = bits[i]
        b = bits[i + 1 :]
        inter = np.logical_and(a, b).sum(axis=1)
        union = np.logical_or(a, b).sum(axis=1)
        valid = union > 0
        sims.extend((inter[valid] / union[valid]).tolist())
    return float(np.median(sims))


def load_smiles_lookup(feature_names):
    if not SMILES_LOOKUP_FILE.exists():
        return pd.DataFrame(columns=["Canonical_SMILES", *feature_names])
    lookup = pd.read_csv(SMILES_LOOKUP_FILE)
    if "SMILES" not in lookup.columns:
        return pd.DataFrame(columns=["Canonical_SMILES", *feature_names])
    missing = [c for c in feature_names if c not in lookup.columns]
    if missing:
        raise ValueError(
            "smiles_descriptor_lookup.csv is missing required descriptor columns: "
            + ", ".join(missing[:10])
        )
    lookup = lookup.copy()
    lookup["Canonical_SMILES"] = lookup["SMILES"].map(canonical_smiles)
    lookup = lookup.dropna(subset=["Canonical_SMILES"])
    lookup = lookup.drop_duplicates("Canonical_SMILES", keep="first")
    return lookup.loc[:, ["Canonical_SMILES", *feature_names]].reset_index(drop=True)


def canonical_smiles(smiles):
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return str(smiles).strip()


if __name__ == "__main__":
    main()
