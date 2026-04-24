"""
Intervener complexity classifiers: LogReg, RandomForest, XGBoost, MLP.
Also: cross-lingual transfer and SHAP analysis.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# AUDIT FIX (2026-04-24): Removed 'arity', 'subtree_size', 'depth' — these are direct
# components of complexity_score = 0.35*arity + 0.25*subtree_size + 0.20*depth + 0.20*POS.
# Using them to predict complexity_score > 1.5 is circular data leakage (Kaufman et al. 2012).
# Features below are INDEPENDENT of the complexity formula — scientifically valid predictors.
FEATURE_COLS = [
    "dependency_distance",   # linear gap between head & dependent — not in formula
    "direction_enc",         # left=0, right=1 dependency — not in formula
    "head_upos_enc",         # POS of head word — not in formula
    "dependent_upos_enc",    # POS of dependent word — not in formula
    "intervener_upos_enc",   # POS of intervener — raw POS, not POS_weight used in formula
    "morph_richness",        # morphological feature count — not in formula
]


def _encode_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """One-hot encode categorical features and return (X, y) arrays."""
    df = df.copy()

    # Encode direction
    df["direction_enc"] = (df["direction"] == "right").astype(int)

    # Encode UPOS columns
    for col in ["head_upos", "dependent_upos", "intervener_upos"]:
        enc_col = col + "_enc"
        le = LabelEncoder()
        df[enc_col] = le.fit_transform(df[col].fillna("X").astype(str))

    # Encode language
    if "language" in df.columns:
        le_lang = LabelEncoder()
        df["language_enc"] = le_lang.fit_transform(df["language"].astype(str))
        feature_cols = FEATURE_COLS + ["language_enc"]
    else:
        feature_cols = FEATURE_COLS

    avail = [c for c in feature_cols if c in df.columns]
    X = df[avail].fillna(0).values.astype(float)
    y = (df["complexity_label"] == "high").astype(int).values if "complexity_label" in df.columns else np.zeros(len(df), dtype=int)
    return X, y


def build_models(cfg: Dict) -> Dict[str, Any]:
    ml_cfg = cfg.get("ml", {})
    seed = cfg.get("project", {}).get("random_seed", 42)

    lr_cfg = ml_cfg.get("models", {}).get("logistic_regression", {})
    rf_cfg = ml_cfg.get("models", {}).get("random_forest", {})
    xgb_cfg = ml_cfg.get("models", {}).get("xgboost", {})
    mlp_cfg = ml_cfg.get("models", {}).get("mlp", {})

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=lr_cfg.get("max_iter", 1000),
                C=lr_cfg.get("C", 1.0),
                random_state=seed,
                n_jobs=ml_cfg.get("n_jobs", -1),
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 100),
            max_depth=rf_cfg.get("max_depth", 10),
            random_state=seed,
            n_jobs=ml_cfg.get("n_jobs", -1),
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 100),
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.1),
            random_state=seed,
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=tuple(mlp_cfg.get("hidden_layer_sizes", [128, 64])),
                max_iter=mlp_cfg.get("max_iter", 200),
                random_state=seed,
            )),
        ]),
    }

    # Try XGBoost if available
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 100),
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.1),
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        )
    except ImportError:
        logger.info("XGBoost not installed; using GradientBoosting instead")

    return models


class IntervenorClassifier:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.seed = cfg.get("project", {}).get("random_seed", 42)
        self.threshold = cfg.get("ml", {}).get("complexity_threshold", 1.5)
        self.cv_folds = cfg.get("ml", {}).get("cv_folds", 5)
        self.models = build_models(cfg)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        df["complexity_label"] = (df["complexity_score"] >= self.threshold).map(
            {True: "high", False: "low"}
        )
        return _encode_features(df)

    def train_evaluate(
        self, language: str, df: pd.DataFrame, max_rows: int = 50_000
    ) -> List[Dict]:
        if len(df) < 50:
            logger.warning("[%s] Too few samples for ML (%d rows)", language, len(df))
            return []

        if len(df) > max_rows:
            logger.info("[%s] Sampling %d / %d rows for ML", language, max_rows, len(df))
            df = df.sample(n=max_rows, random_state=self.seed)

        X, y = self.prepare_data(df)
        if y.sum() == 0 or (len(y) - y.sum()) == 0:
            logger.warning("[%s] Single class — skipping ML", language)
            return []

        results = []
        cv = StratifiedKFold(n_splits=min(self.cv_folds, 5), shuffle=True, random_state=self.seed)

        for name, model in self.models.items():
            try:
                scores = cross_validate(
                    model, X, y, cv=cv,
                    scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
                    n_jobs=1,
                )
                results.append({
                    "language": language,
                    "model_name": name,
                    "accuracy": float(np.mean(scores["test_accuracy"])),
                    "precision": float(np.mean(scores["test_precision_weighted"])),
                    "recall": float(np.mean(scores["test_recall_weighted"])),
                    "f1_score": float(np.mean(scores["test_f1_weighted"])),
                })
                logger.info("[%s] %s — F1: %.3f", language, name,
                            results[-1]["f1_score"])
            except Exception as e:
                logger.warning("[%s] %s failed: %s", language, name, e)

        return results

    def feature_importance(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Return feature importances from RandomForest."""
        if len(df) < 50:
            return None
        X, y = self.prepare_data(df)
        rf = self.models.get("RandomForest")
        if rf is None:
            return None
        try:
            rf.fit(X, y)
            avail = [c for c in FEATURE_COLS + ["language_enc"] if c in df.columns or c in FEATURE_COLS]
            importances = pd.DataFrame({
                "feature": avail[:X.shape[1]],
                "importance": rf.feature_importances_,
            }).sort_values("importance", ascending=False)
            return importances
        except Exception as e:
            logger.warning("Feature importance failed: %s", e)
            return None

    def shap_analysis(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """SHAP values from RandomForest or XGBoost (if shap is installed)."""
        try:
            import shap
        except ImportError:
            logger.info("shap not installed; skipping SHAP analysis")
            return None

        if len(df) < 50:
            return None
        X, y = self.prepare_data(df)
        model = self.models.get("XGBoost") or self.models.get("GradientBoosting")
        if model is None:
            return None
        try:
            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            avail = [c for c in FEATURE_COLS + ["language_enc"]][:X.shape[1]]
            mean_shap = pd.DataFrame({
                "feature": avail,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)
            return mean_shap
        except Exception as e:
            logger.warning("SHAP analysis failed: %s", e)
            return None

    def cross_lingual_transfer(
        self,
        train_dfs: Dict[str, pd.DataFrame],
        test_lang: str,
        test_df: pd.DataFrame,
    ) -> List[Dict]:
        """Train on source languages, evaluate on target language."""
        results = []
        if len(test_df) < 20:
            return results

        X_test, y_test = self.prepare_data(test_df)
        if y_test.sum() == 0:
            return results

        for src_lang, src_df in train_dfs.items():
            if src_lang == test_lang or len(src_df) < 50:
                continue
            X_train, y_train = self.prepare_data(src_df)
            if y_train.sum() == 0:
                continue

            for name in ["RandomForest", "GradientBoosting"]:
                model = self.models.get(name)
                if model is None:
                    continue
                try:
                    model.fit(X_train, y_train)
                    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                    y_pred = model.predict(X_test)
                    results.append({
                        "source_language": src_lang,
                        "target_language": test_lang,
                        "model_name": name,
                        "accuracy": accuracy_score(y_test, y_pred),
                        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    })
                except Exception as e:
                    logger.warning("Cross-lingual %s->%s %s failed: %s",
                                   src_lang, test_lang, name, e)
        return results
