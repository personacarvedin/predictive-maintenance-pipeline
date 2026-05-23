# tests/test_preprocessing.py

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_COLUMN


def make_sample_df(n=50, seed=42):
    """Generate a minimal synthetic dataframe that mirrors the real schema."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "UDI"                         : range(1, n + 1),
        "Product ID"                  : [f"M{i}" for i in range(n)],
        "Type"                        : rng.choice(["L", "M", "H"], size=n),
        "Air temperature [K]"         : rng.normal(300, 2, n),
        "Process temperature [K]"     : rng.normal(310, 2, n),
        "Rotational speed [rpm]"      : rng.normal(1500, 100, n),
        "Torque [Nm]"                 : rng.normal(40, 5, n),
        "Tool wear [min]"             : rng.integers(0, 250, n).astype(float),
        "Machine failure"             : rng.choice([0, 1], size=n, p=[0.97, 0.03]),
        "TWF"                         : rng.integers(0, 2, n),
        "HDF"                         : rng.integers(0, 2, n),
        "PWF"                         : rng.integers(0, 2, n),
        "OSF"                         : rng.integers(0, 2, n),
        "RNF"                         : rng.integers(0, 2, n),
    })


# ─────────────────────────────────────────────────────────────────────────────
class TestPreprocess:

    def test_drop_irrelevant_columns(self, tmp_path, monkeypatch):
        """Columns listed in DROP_COLUMNS must be removed."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True)

        import config as cfg
        monkeypatch.setattr(cfg, "PROCESSED_DATA_DIR", str(tmp_path / "data" / "processed"))
        monkeypatch.setattr(cfg, "PROCESSED_DATA_PATH",
                            str(tmp_path / "data" / "processed" / "processed_data.csv"))

        from src.preprocessing import preprocess
        df = make_sample_df()
        result = preprocess(df)

        for col in cfg.DROP_COLUMNS:
            assert col not in result.columns, f"'{col}' should have been dropped"

    def test_no_missing_values_after_preprocess(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True)

        import config as cfg
        monkeypatch.setattr(cfg, "PROCESSED_DATA_DIR", str(tmp_path / "data" / "processed"))
        monkeypatch.setattr(cfg, "PROCESSED_DATA_PATH",
                            str(tmp_path / "data" / "processed" / "processed_data.csv"))

        from src.preprocessing import preprocess
        df = make_sample_df()
        # Inject some NaNs
        df.loc[0, "Torque [Nm]"] = np.nan
        result = preprocess(df)
        assert result.isnull().sum().sum() == 0

    def test_type_column_is_numeric(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True)

        import config as cfg
        monkeypatch.setattr(cfg, "PROCESSED_DATA_DIR", str(tmp_path / "data" / "processed"))
        monkeypatch.setattr(cfg, "PROCESSED_DATA_PATH",
                            str(tmp_path / "data" / "processed" / "processed_data.csv"))

        from src.preprocessing import preprocess
        df = make_sample_df()
        result = preprocess(df)
        assert pd.api.types.is_numeric_dtype(result["Type"])

    def test_target_column_preserved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True)

        import config as cfg
        monkeypatch.setattr(cfg, "PROCESSED_DATA_DIR", str(tmp_path / "data" / "processed"))
        monkeypatch.setattr(cfg, "PROCESSED_DATA_PATH",
                            str(tmp_path / "data" / "processed" / "processed_data.csv"))

        from src.preprocessing import preprocess
        df = make_sample_df()
        result = preprocess(df)
        assert TARGET_COLUMN in result.columns


class TestFeatureEngineering:

    def _get_processed(self):
        return pd.DataFrame({
            "Type"                        : [0, 1, 2, 0, 1],
            "Air temperature [K]"         : [-0.5, 0.2, 1.1, -0.3, 0.8],
            "Process temperature [K]"     : [-0.4, 0.3, 1.2, -0.2, 0.9],
            "Rotational speed [rpm]"      : [0.1, -0.5, 1.0,  0.2, -0.3],
            "Torque [Nm]"                 : [-0.2, 0.4, 1.5, -0.1, 0.6],
            "Tool wear [min]"             : [-1.0, 0.0, 2.0,  0.5, 1.0],
            "Machine failure"             : [0, 0, 1, 0, 1],
        })

    def test_split_sizes(self):
        from src.feature_engineering import engineer_features
        df = self._get_processed()
        X_train, X_test, y_train, y_test, _ = engineer_features(df)
        assert len(X_train) + len(X_test) == len(df)

    def test_derived_features_created(self):
        from src.feature_engineering import engineer_features
        df = self._get_processed()
        _, _, _, _, feature_names = engineer_features(df)
        assert "temperature_diff" in feature_names
        assert "power_proxy" in feature_names

    def test_target_not_in_features(self):
        from src.feature_engineering import engineer_features
        df = self._get_processed()
        X_train, X_test, _, _, feature_names = engineer_features(df)
        assert TARGET_COLUMN not in feature_names
        assert TARGET_COLUMN not in X_train.columns