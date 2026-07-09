import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def empty_feature_row(feature_names):
    return pd.DataFrame([{name: 0 for name in feature_names}])


def align_descriptor_frame(df, feature_names):
    aligned = df.copy()
    missing = [c for c in feature_names if c not in aligned.columns]
    for col in missing:
        aligned[col] = 0
    return aligned.loc[:, feature_names].fillna(0)


ROOT = Path(__file__).resolve().parent
PADEL_CONFIG = ROOT / "padel_apc_pubchem.xml"
PADEL_DIR = ROOT / "padel"
PADEL_JAR = PADEL_DIR / "PaDEL-Descriptor.jar"


def descriptors_from_smiles(smiles, feature_names):
    row = empty_feature_row(feature_names)
    mol = _mol_from_smiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES. Please check the molecule.")

    _fill_rdkit_topological_torsion(row, mol, feature_names)
    _fill_padel_descriptors(row, smiles, feature_names)
    return row


def _mol_from_smiles(smiles):
    try:
        from rdkit import Chem
    except Exception as exc:
        raise RuntimeError("RDKit is not installed. Install requirements.txt first.") from exc
    return Chem.MolFromSmiles(smiles)


def _fill_rdkit_topological_torsion(row, mol, feature_names):
    try:
        from rdkit.Chem import rdMolDescriptors
    except Exception:
        return

    fp_cols = [c for c in feature_names if c.startswith("FP_")]
    if not fp_cols:
        return
    max_bit = max(int(c.split("_", 1)[1]) for c in fp_cols)
    n_bits = max(2048, max_bit + 1)
    bitvect = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
        mol, nBits=n_bits
    )
    on_bits = set(bitvect.GetOnBits())
    for col in fp_cols:
        bit = int(col.split("_", 1)[1])
        row.loc[0, col] = 1 if bit in on_bits else 0


def _fill_padel_descriptors(row, smiles, feature_names):
    padel_cols = [
        c for c in feature_names if c.startswith("APC2D") or c.startswith("PubchemFP")
    ]
    if not padel_cols:
        return
    values = _run_padel_apc_pubchem(smiles)

    for col in padel_cols:
        candidates = _padel_name_candidates(col)
        for name in candidates:
            if name in values:
                row.loc[0, col] = _coerce_number(values[name])
                break


def _padel_name_candidates(col):
    names = [col]
    if col.startswith("PubchemFP"):
        bit = re.sub(r"\D", "", col)
        names.extend([f"PubchemFP{bit}", f"PubchemFP{int(bit)}"])
    return names


def _coerce_number(value):
    try:
        if value in ("", "NaN", None):
            return 0
        number = float(value)
        if np.isnan(number):
            return 0
        return number
    except Exception:
        return 0


def _run_padel_apc_pubchem(smiles):
    if not PADEL_JAR.exists():
        raise RuntimeError("Bundled PaDEL-Descriptor.jar is missing from the padel folder.")

    work_dir = Path(tempfile.mkdtemp(prefix="keap_padel_"))
    mol_dir = work_dir / "mols"
    mol_dir.mkdir()
    smi_file = mol_dir / "query.smi"
    out_file = work_dir / "descriptors.csv"
    smi_file.write_text(f"{smiles} query\n", encoding="utf-8")

    try:
        command = [
            "java",
            "-jar",
            str(PADEL_JAR),
            "-dir",
            str(mol_dir),
            "-file",
            str(out_file),
            "-descriptortypes",
            str(PADEL_CONFIG),
            "-fingerprints",
            "-2d",
            "-removesalt",
            "-detectaromaticity",
            "-standardizenitro",
            "-threads",
            "1",
            "-maxruntime",
            "-1",
        ]
        completed = subprocess.run(
            command,
            cwd=str(PADEL_DIR),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0 and (not out_file.exists() or out_file.stat().st_size == 0):
            raise RuntimeError(
                "PaDEL descriptor calculation failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )

        df = pd.read_csv(out_file)
        if df.empty:
            raise RuntimeError("PaDEL descriptor calculation returned no rows.")
        return df.iloc[0].to_dict()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
