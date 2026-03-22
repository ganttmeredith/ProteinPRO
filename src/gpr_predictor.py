"""
GPR (Gaussian Process Regression) stability predictor with uncertainty estimates.

Uses sklearn's GaussianProcessRegressor for compatibility. Provides the same
interface as StabilityPredictor but returns epistemic uncertainty (std dev)
alongside the predicted score. Inspired by Quinn's Bayesian Optimization approach.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler

from .pdb_handler import featurize_protein, load_config
from .monomer_featurizer import (
    featurize_all_monomers,
    composition_to_polymer_features,
)
from .stability_model import (
    build_feature_vector,
    sample_design_space,
    PROTEIN_FEAT_KEYS,
    POLYMER_FEAT_KEYS,
)


class GPRStabilityPredictor:
    """
    GPR-based stability predictor with uncertainty estimates.
    Uses same feature space as StabilityPredictor; trained on physics-informed surrogate.
    """

    def __init__(self, kernel="matern", nu=2.5, random_state=42):
        self.scaler = StandardScaler()
        # Matern kernel (like Quinn's setup) - smooth, works well for chemical descriptors
        k = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=nu)
        self.gpr = GaussianProcessRegressor(
            kernel=k,
            n_restarts_optimizer=5,
            random_state=random_state,
            alpha=1e-6,  # numerical stability
        )
        self.feature_names = PROTEIN_FEAT_KEYS + POLYMER_FEAT_KEYS
        self.monomer_features_df: Optional[pd.DataFrame] = None
        self._fitted = False

    def _init_surrogate(self):
        """Train GPR on physics-informed surrogate data (same as RF surrogate)."""
        self.monomer_features_df = featurize_all_monomers()
        X_demo = []
        np.random.seed(42)
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
        y_demo = (
            0.3 * (1 - np.abs(X_demo[:, 0] - X_demo[:, 7]))
            + 0.3 * (1 - np.abs(X_demo[:, 6]))
            + 0.2 * np.clip(X_demo[:, 2] + X_demo[:, 3], 0, 0.3)
            + 0.2 * np.random.randn(len(X_demo)) * 0.1
        )
        X_scaled = self.scaler.fit_transform(X_demo)
        self.gpr.fit(X_scaled, y_demo)
        self._fitted = True

    def predict(
        self,
        protein_source: Union[str, Path],
        composition: Dict[str, float],
        pdb_id: Optional[str] = None,
    ) -> Tuple[float, dict]:
        """
        Predict stability score with uncertainty.
        Returns (score, details_dict) where details includes "uncertainty".
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
        raw_mean, raw_std = self.gpr.predict(vec_scaled, return_std=True)
        raw_score = float(raw_mean[0])
        uncertainty = float(raw_std[0])
        score = float((np.tanh(raw_score / 50) + 1) / 2)
        # Scale uncertainty to ~0-1 for display (tanh derivative approximation)
        uncertainty_scaled = float(uncertainty / 50 * 0.5)  # rough calibration
        details = {
            "protein_features": {k: pf.get(k) for k in PROTEIN_FEAT_KEYS},
            "polymer_features": poly_f,
            "raw_score": raw_score,
            "uncertainty": uncertainty,
            "uncertainty_scaled": min(uncertainty_scaled, 0.5),  # cap for display
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
        """Sample design space and rank by predicted stability."""
        compositions = sample_design_space(n_candidates)
        results = self.predict_batch(protein_source, compositions, pdb_id)
        rows = []
        for comp, (score, details) in zip(compositions, results):
            row = {
                "composition": comp,
                "stability_score": score,
                "uncertainty": details.get("uncertainty_scaled", details.get("uncertainty", 0)),
            }
            row.update(details.get("protein_features", {}))
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.sort_values("stability_score", ascending=False).reset_index(drop=True)
        return df
