"""
Chemical descriptor featurization for PET-RAFT compatible monomers.

Uses RDKit and Mordred for comprehensive molecular descriptors.
Falls back to minimal descriptor set if Mordred unavailable.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from mordred import Calculator, descriptors as mordred_descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False


def load_monomers(config_path: Optional[Path] = None) -> dict:
    """Load monomer SMILES and metadata from config."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("monomers", {})


def _rdkit_basic_descriptors(mol) -> dict:
    """Core RDKit descriptors (always available when RDKit present)."""
    if mol is None:
        return {}
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "NumSaturatedRings": Descriptors.NumSaturatedRings(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
    }


def _mordred_extended_descriptors(mol) -> dict:
    """Extended Mordred descriptors when available."""
    if not MORDRED_AVAILABLE or mol is None:
        return {}
    calc = Calculator(
        mordred_descriptors.ABCIndex,
        mordred_descriptors.AcidBase,
        mordred_descriptors.Aromatic,
        mordred_descriptors.Ring,
        mordred_descriptors.AtomCount,
        mordred_descriptors.BondCount,
    )
    try:
        res = calc(mol)
        return {str(k): v for k, v in res.asdict().items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    except Exception:
        return {}


def featurize_monomer(smiles: str, name: str = "") -> dict:
    """
    Compute chemical descriptors for a single monomer.
    Returns dict with all descriptors, normalized for ML.
    """
    out = {"monomer_name": name, "smiles": smiles}
    if not RDKIT_AVAILABLE:
        return out

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out

    basic = _rdkit_basic_descriptors(mol)
    out.update(basic)
    extended = _mordred_extended_descriptors(mol)
    out.update(extended)
    return out


def featurize_all_monomers(config_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Featurize all monomers from config.
    Returns DataFrame with one row per monomer.
    """
    monomers = load_monomers(config_path)
    rows = []
    for name, info in monomers.items():
        smiles = info.get("smiles", "")
        row = featurize_monomer(smiles, name)
        row["category"] = info.get("category", "unknown")
        rows.append(row)
    return pd.DataFrame(rows)


def composition_to_polymer_features(
    composition: Dict[str, float],
    monomer_features_df: pd.DataFrame,
    descriptor_cols: Optional[List[str]] = None,
) -> dict:
    """
    Given a monomer composition (name -> molar fraction), compute
    weighted-averaged polymer-level descriptors.
    """
    if descriptor_cols is None:
        # Standard numeric descriptor columns
        numeric_cols = monomer_features_df.select_dtypes(include=[np.number]).columns.tolist()
        descriptor_cols = [c for c in numeric_cols if c != "monomer_name"]

    weighted = {}
    total_frac = sum(composition.values())
    if total_frac < 1e-6:
        return weighted

    for name, frac in composition.items():
        if frac <= 0:
            continue
        row = monomer_features_df[monomer_features_df["monomer_name"] == name]
        if row.empty:
            continue
        w = frac / total_frac
        for col in descriptor_cols:
            if col in row.columns:
                val = row[col].iloc[0]
                if isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
                    weighted[col] = weighted.get(col, 0) + w * val
    return weighted
