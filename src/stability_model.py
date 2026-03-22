"""
Stability prediction model for polymer-protein hybrid formulation.

Combines protein features (from PDB) and polymer features (from monomer
composition + chemical descriptors) to predict formulation stability.
Designed for interpretability and publication-ready methodology.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .pdb_handler import featurize_protein, load_config
from .monomer_featurizer import (
    featurize_all_monomers,
    composition_to_polymer_features,
    load_monomers,
)


# Feature names used in the model (subset for reproducibility)
PROTEIN_FEAT_KEYS = [
    "mean_hydrophobicity",
    "std_hydrophobicity",
    "fraction_positive",
    "fraction_negative",
    "fraction_hydrophobic",
    "fraction_polar",
    "net_charge_density",
]

POLYMER_FEAT_KEYS = [
    "MolWt",
    "LogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "FractionCSP3",
]

# Wrapper for LogisticRegression: bins target, predicts P(high stability)
class _LogisticRegressionRegressor:
    """Uses LogisticRegression on binned targets; predict returns scaled score for tanh pipeline."""

    def __init__(self, **kwargs):
        self.clf = LogisticRegression(**kwargs, random_state=42)
        self._median = None

    def fit(self, X, y):
        self._median = np.median(y)
        y_bin = (y >= self._median).astype(int)
        self.clf.fit(X, y_bin)
        return self

    def predict(self, X):
        proba = self.clf.predict_proba(X)[:, 1]  # P(high)
        # Scale to ~[0,50] so tanh(raw/50) maps to reasonable [0,1]
        return proba * 50


MODEL_TYPES = {
    "rf": ("Random Forest", RandomForestRegressor, {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 2, "random_state": 42}),
    "svr": ("SVM (SVR)", SVR, {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}),
    "ridge": ("Ridge (linear)", Ridge, {"alpha": 1.0, "random_state": 42}),
    "logistic": ("Logistic Regression", _LogisticRegressionRegressor, {}),
    "gradient_boosting": ("Gradient Boosting", GradientBoostingRegressor, {"n_estimators": 50, "max_depth": 4, "random_state": 42}),
    "knn": ("K-Nearest Neighbors", KNeighborsRegressor, {"n_neighbors": 5}),
}


def build_feature_vector(
    protein_features: dict,
    polymer_features: dict,
) -> np.ndarray:
    """
    Build combined feature vector from protein and polymer descriptors.
    """
    p_vec = [protein_features.get(k, 0) for k in PROTEIN_FEAT_KEYS]
    poly_vec = [polymer_features.get(k, 0) for k in POLYMER_FEAT_KEYS]
    return np.array(p_vec + poly_vec, dtype=np.float64)


def sample_design_space(
    n_samples: int = 100,
    config_path: Optional[Path] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    """
    Sample monomer compositions within design space constraints.
    Returns list of {monomer_name: fraction} dicts.
    """
    rng = np.random.default_rng(seed)
    config = load_config(config_path)
    monomers = list(load_monomers(config_path).keys())
    ds = config.get("design_space", {})
    min_frac = ds.get("min_monomer_fraction", 0.05)
    max_frac = ds.get("max_monomer_fraction", 0.50)

    compositions = []
    for _ in range(n_samples):
        # Random number of monomers (2-5 typically)
        n_monomers = rng.integers(2, min(6, len(monomers) + 1))
        selected = rng.choice(monomers, size=n_monomers, replace=False)
        raw = rng.uniform(min_frac, max_frac, size=n_monomers)
        raw /= raw.sum()
        comp = dict(zip(selected, raw))
        compositions.append(comp)
    return compositions


class StabilityPredictor:
    """
    Predicts stability score for polymer-protein hybrid formulations.
    Can be trained on labeled data or used with a pre-trained surrogate.
    Supports multiple model types: rf, svr, ridge, gradient_boosting, knn.
    """

    def __init__(
        self,
        use_surrogate: bool = True,
        model_path: Optional[Union[str, Path]] = None,
        model_type: str = "rf",
    ):
        self.scaler = StandardScaler()
        self.model_type = model_type if model_type in MODEL_TYPES else "rf"
        name, model_cls, kwargs = MODEL_TYPES[self.model_type]
        self.model = model_cls(**kwargs)
        self.feature_names = PROTEIN_FEAT_KEYS + POLYMER_FEAT_KEYS
        self.monomer_features_df: Optional[pd.DataFrame] = None
        self._fitted = False

        if model_path and Path(model_path).exists():
            self.load(model_path)
        elif use_surrogate:
            self._init_surrogate()

    def _init_surrogate(self):
        """
        Initialize with physics-informed surrogate: hydrophobic matching
        and charge complementarity as proxies for stability.
        """
        self.monomer_features_df = featurize_all_monomers()
        # Fit scaler on representative samples
        X_demo = []
        for _ in range(200):
            pf = {
                "mean_hydrophobicity": np.random.uniform(-1, 2),
                "std_hydrophobicity": np.random.uniform(0.5, 2),
                "fraction_positive": np.random.uniform(0, 0.2),
                "fraction_negative": np.random.uniform(0, 0.2),
                "fraction_hydrophobic": np.random.uniform(0.2, 0.6),
                "fraction_polar": np.random.uniform(0.1, 0.4),
                "net_charge_density": np.random.uniform(-0.2, 0.2),
            }
            comp = sample_design_space(n_samples=1, seed=np.random.randint(1e6))[0]
            poly_f = composition_to_polymer_features(comp, self.monomer_features_df)
            vec = build_feature_vector(pf, poly_f)
            X_demo.append(vec)
        X_demo = np.array(X_demo)
        # Surrogate target: higher when hydrophobicity matches, charge balanced
        y_demo = (
            0.3 * (1 - np.abs(X_demo[:, 0] - X_demo[:, 7]))  # hydrophobicity match
            + 0.3 * (1 - np.abs(X_demo[:, 6]))  # neutral charge
            + 0.2 * np.clip(X_demo[:, 2] + X_demo[:, 3], 0, 0.3)  # some polarity
            + 0.2 * np.random.randn(len(X_demo)) * 0.1
        )
        X_scaled = self.scaler.fit_transform(X_demo)
        self.model.fit(X_scaled, y_demo)
        self._fitted = True

    def predict(
        self,
        protein_source: Union[str, Path],
        composition: Dict[str, float],
        pdb_id: Optional[str] = None,
    ) -> Tuple[float, dict]:
        """
        Predict stability score for a protein + monomer composition.
        Returns (score, details_dict).
        """
        if not self._fitted:
            self._init_surrogate()
        if self.monomer_features_df is None:
            self.monomer_features_df = featurize_all_monomers()

        pf = featurize_protein(protein_source, pdb_id)
        poly_f = composition_to_polymer_features(
            composition,
            self.monomer_features_df,
            descriptor_cols=POLYMER_FEAT_KEYS,
        )
        vec = build_feature_vector(pf, poly_f).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)
        raw_score = float(self.model.predict(vec_scaled)[0])
        # Calibrate to 0-1 range for interpretability (surrogate-specific)
        # Raw scores can vary widely; scale for reasonable spread
        score = float((np.tanh(raw_score / 50) + 1) / 2)
        details = {
            "protein_features": {k: pf.get(k) for k in PROTEIN_FEAT_KEYS},
            "polymer_features": poly_f,
            "raw_score": raw_score,
        }
        return score, details

    def predict_batch(
        self,
        protein_source: Union[str, Path],
        compositions: List[Dict[str, float]],
        pdb_id: Optional[str] = None,
    ) -> List[Tuple[float, dict]]:
        """Predict for multiple compositions."""
        return [
            self.predict(protein_source, comp, pdb_id)
            for comp in compositions
        ]

    def rank_formulations(
        self,
        protein_source: Union[str, Path],
        n_candidates: int = 50,
        pdb_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Sample design space and rank formulations by predicted stability.
        """
        compositions = sample_design_space(n_candidates)
        results = self.predict_batch(protein_source, compositions, pdb_id)
        rows = []
        for comp, (score, details) in zip(compositions, results):
            row = {"composition": comp, "stability_score": score}
            row.update(details.get("protein_features", {}))
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.sort_values("stability_score", ascending=False).reset_index(drop=True)
        return df

    def save(self, path: Union[str, Path]):
        """Save model and scaler."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Union[str, Path]):
        """Load model and scaler."""
        with open(Path(path), "rb") as f:
            state = pickle.load(f)
        self.model = state["model"]
        self.scaler = state["scaler"]
        self.feature_names = state.get("feature_names", self.feature_names)
        self._fitted = True
