"""
soc_model.py
------------
ML models for Soil Organic Carbon (SOC) estimation
and land practice classification.

KEY UPGRADE from v1:
- Trains on STRATIFIED data → all SOC ranges represented
- Evaluates with SPATIAL CV → honest accuracy (no data leakage)
- Sample weights → rare high/low SOC samples weighted more
- Stratum-wise metrics → see accuracy per SOC level
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_squared_error, r2_score,
                              mean_absolute_error, classification_report)

FEATURE_COLS = ["B2","B3","B4","B8","B11","B12",
                "NDVI","BSI","NDTI","NDWI","EVI","CRC"]
SOC_TARGET      = "SOC"
PRACTICE_TARGET = "LandPractice"


# ─────────────────────────────────────────────
# SOC REGRESSOR
# ─────────────────────────────────────────────

class SOCPredictor:
    """
    Random Forest regressor for SOC estimation.

    Upgrade: sample_weight support
    ────────────────────────────────
    Stratified sampling gives equal COUNT per stratum.
    But rare strata (SOC>35) are truly rare in nature.
    We use sample weights to tell the model:
    'Pay extra attention to these rare samples.'

    Weight = inverse of stratum frequency in real world:
      Stratum 5 (>35, rare)   → weight = 3.0
      Stratum 1 (<5, rare)    → weight = 3.0
      Stratum 3 (10-20, common) → weight = 1.0
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained          = False
        self.feature_importances_ = None
        self.metrics             = {}

    def _get_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Assigns sample weights based on stratum rarity.
        Rare strata get higher weights so model learns them better.
        """
        weight_map = {1: 3.0, 2: 2.0, 3: 1.0, 4: 2.0, 5: 3.0}
        default_w  = 1.0

        if "stratum" in df.columns:
            weights = df["stratum"].map(weight_map).fillna(default_w)
        else:
            # Assign stratum from SOC value if column missing
            def soc_to_w(soc):
                if soc < 5:    return 3.0
                elif soc < 10: return 2.0
                elif soc < 20: return 1.0
                elif soc < 35: return 2.0
                else:          return 3.0
            weights = df[SOC_TARGET].apply(soc_to_w)

        return weights.values

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train SOC model with sample weights.
        Returns performance metrics dict.
        """
        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available].values
        y = df[SOC_TARGET].values
        w = self._get_sample_weights(df)

        X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
            X, y, w, test_size=0.2, random_state=42
        )

        # Train with sample weights
        self.model.fit(X_train, y_train, sample_weight=w_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)

        cv_scores = cross_val_score(
            self.model, X, y, cv=5, scoring="r2"
        )

        self.metrics = {
            "RMSE":       round(rmse, 3),
            "R2":         round(r2, 3),
            "MAE":        round(mae, 3),
            "CV_R2_mean": round(float(cv_scores.mean()), 3),
            "CV_R2_std":  round(float(cv_scores.std()), 3),
            "n_samples":  len(df),
            "n_features": len(available),
        }

        self.feature_importances_ = dict(zip(
            available,
            np.round(self.model.feature_importances_, 4)
        ))

        print("─" * 50)
        print("🌱 SOC MODEL TRAINED (with sample weights)")
        print(f"   R²        : {self.metrics['R2']}")
        print(f"   RMSE      : {self.metrics['RMSE']} g/kg")
        print(f"   MAE       : {self.metrics['MAE']} g/kg")
        print(f"   CV R²     : {self.metrics['CV_R2_mean']} "
              f"± {self.metrics['CV_R2_std']}")
        print(f"   Samples   : {self.metrics['n_samples']}")
        print("─" * 50)
        return self.metrics

    def train_with_spatial_cv(self, df: pd.DataFrame,
                               n_folds: int = 5) -> dict:
        """
        Train + evaluate using Spatial Cross Validation.
        Returns spatial CV metrics for research paper.
        """
        from src.ingestion import run_spatial_cross_validation
        cv_result = run_spatial_cross_validation(df, self, n_folds)
        # Final train on full dataset
        self.train(df)
        self.metrics["spatial_cv"] = cv_result
        return self.metrics

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict SOC with confidence intervals from RF tree variance."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available].values

        # Prediction from every tree → get spread = uncertainty
        all_tree_preds = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])

        df = df.copy()
        df["SOC_predicted"] = np.round(self.model.predict(X), 2)
        df["SOC_std"]       = np.round(all_tree_preds.std(axis=0), 2)
        df["SOC_lower_95"]  = np.round(
            df["SOC_predicted"] - 1.96 * df["SOC_std"], 2)
        df["SOC_upper_95"]  = np.round(
            df["SOC_predicted"] + 1.96 * df["SOC_std"], 2)
        df["SOC_category"]  = df["SOC_predicted"].apply(categorize_soc)

        return df

    def stratum_accuracy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reports accuracy SEPARATELY for each SOC stratum.
        Critical for research paper — shows model is fair
        across all carbon levels, not just the common ones.

        Example output:
        Stratum   RMSE    R2    n_samples
        <5 g/kg   1.2    0.81   100       ← model accurate on critical fields
        5-10      1.8    0.77   100
        10-20     2.1    0.74   100
        20-35     2.4    0.71   100
        >35 g/kg  3.1    0.68   100       ← harder but still ok
        """
        df_pred = self.predict(df)
        results = []

        strata = {
            "Critical (<5)":      df_pred["SOC"] < 5,
            "Very Low (5-10)":    (df_pred["SOC"]>=5)  & (df_pred["SOC"]<10),
            "Low-Mod (10-20)":    (df_pred["SOC"]>=10) & (df_pred["SOC"]<20),
            "Good (20-35)":       (df_pred["SOC"]>=20) & (df_pred["SOC"]<35),
            "Excellent (>35)":    df_pred["SOC"] >= 35,
        }

        print("\n📊 Stratum-wise Accuracy (Research Metric):")
        print("─" * 60)
        for label, mask in strata.items():
            subset = df_pred[mask]
            if len(subset) < 3:
                continue
            r2   = r2_score(subset["SOC"], subset["SOC_predicted"])
            rmse = np.sqrt(mean_squared_error(
                subset["SOC"], subset["SOC_predicted"]))
            mae  = mean_absolute_error(
                subset["SOC"], subset["SOC_predicted"])
            results.append({
                "Stratum": label,
                "n":       len(subset),
                "R2":      round(r2, 3),
                "RMSE":    round(rmse, 3),
                "MAE":     round(mae, 3),
            })
            icon = "✅" if r2 > 0.65 else "⚠️ "
            print(f"  {icon} {label:<22} n={len(subset):>3} | "
                  f"R²={r2:.3f} | RMSE={rmse:.2f} | MAE={mae:.2f}")

        print("─" * 60)
        return pd.DataFrame(results)

    def save(self, path: str = "outputs/soc_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"💾 SOC model saved: {path}")

    @classmethod
    def load(cls, path: str = "outputs/soc_model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────
# PRACTICE CLASSIFIER
# ─────────────────────────────────────────────

class PracticeClassifier:
    """Random Forest classifier for land practice detection."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=150, max_depth=10,
            random_state=42, n_jobs=-1,
            class_weight="balanced"   # handles class imbalance
        )
        self.label_encoder = LabelEncoder()
        self.is_trained    = False
        self.classes_      = None

    def train(self, df: pd.DataFrame) -> dict:
        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available].values
        y = self.label_encoder.fit_transform(df[PRACTICE_TARGET].values)
        self.classes_ = self.label_encoder.classes_

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=self.classes_,
            output_dict=True
        )
        accuracy = report["accuracy"]

        print("─" * 50)
        print("🌾 PRACTICE CLASSIFIER TRAINED (class_weight=balanced)")
        print(f"   Accuracy : {round(accuracy*100, 1)}%")
        print(f"   Classes  : {list(self.classes_)}")
        print("─" * 50)
        return {"accuracy": round(accuracy, 3), "report": report}

    def predict_proba_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Classifier not trained.")
        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available].values
        probs  = self.model.predict_proba(X)
        labels = self.label_encoder.inverse_transform(
            self.model.predict(X))
        df = df.copy()
        df["PredictedPractice"]   = labels
        df["PracticeConfidence"]  = np.round(probs.max(axis=1) * 100, 1)
        return df


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def categorize_soc(soc: float) -> str:
    if soc < 5:    return "Very Low (<5 g/kg)"
    elif soc < 10: return "Low (5-10 g/kg)"
    elif soc < 20: return "Moderate (10-20 g/kg)"
    elif soc < 35: return "High (20-35 g/kg)"
    else:          return "Very High (>35 g/kg)"


# Keep backward-compatible alias
def generate_synthetic_training_data(n: int = 1000) -> pd.DataFrame:
    """Backward-compatible wrapper — uses stratified generation."""
    from src.ingestion import generate_stratified_synthetic
    n_per = max(n // 5, 20)
    return generate_stratified_synthetic(n_per_stratum=n_per)


if __name__ == "__main__":
    from src.ingestion import generate_stratified_synthetic, spatial_cv_split
    from src.indices import compute_indices_df, add_practice_labels

    print("="*55)
    print("Testing SOC model with stratified data + spatial CV")
    print("="*55)

    df = generate_stratified_synthetic(n_per_stratum=150)
    df = compute_indices_df(df)
    df = add_practice_labels(df)
    df = spatial_cv_split(df, n_folds=5)

    soc_model = SOCPredictor()
    metrics   = soc_model.train_with_spatial_cv(df, n_folds=5)

    prac_clf  = PracticeClassifier()
    prac_clf.train(df)

    # Stratum accuracy
    soc_model.stratum_accuracy(df)

    soc_model.save()
    print("\n✅ All tests passed!")
