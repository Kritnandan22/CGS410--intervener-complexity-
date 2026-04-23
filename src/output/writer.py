"""
Write all standardized CSV outputs, enforcing column schemas.
"""
from __future__ import annotations

import os
import logging
from typing import Dict, List

import pandas as pd

from src.data.validator import SchemaValidator, SCHEMAS

logger = logging.getLogger(__name__)


class OutputWriter:
    def __init__(self, output_root: str):
        self.output_root = output_root
        self.validator = SchemaValidator()

    def _lang_dir(self, language: str) -> str:
        d = os.path.join(self.output_root, language)
        os.makedirs(d, exist_ok=True)
        return d

    def _write(self, df: pd.DataFrame, path: str):
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info("Written: %s (%d rows)", path, len(df))

    def write_intervener_features(self, df: pd.DataFrame, language: str):
        schema = "intervener_features"
        df = self.validator.ensure_columns(df, schema)
        self.validator.validate(df, schema, language)
        path = os.path.join(self._lang_dir(language), "intervener_features.csv")
        self._write(df, path)

    def write_language_summary(self, df: pd.DataFrame, language: str):
        schema = "language_summary"
        df = self.validator.ensure_columns(df, schema)
        self.validator.validate(df, schema, language)
        path = os.path.join(self._lang_dir(language), "language_summary.csv")
        self._write(df, path)

    def write_distribution_data(self, df: pd.DataFrame, language: str):
        schema = "distribution_data"
        df = self.validator.ensure_columns(df, schema)
        self.validator.validate(df, schema, language)
        path = os.path.join(self._lang_dir(language), "distribution_data.csv")
        self._write(df, path)

    def write_ml_results(self, df: pd.DataFrame, language: str):
        schema = "ml_results"
        df = self.validator.ensure_columns(df, schema)
        self.validator.validate(df, schema, language)
        path = os.path.join(self._lang_dir(language), "ml_results.csv")
        self._write(df, path)

    def write_zscore_results(self, df: pd.DataFrame, language: str):
        schema = "zscore_results"
        df = self.validator.ensure_columns(df, schema)
        self.validator.validate(df, schema, language)
        path = os.path.join(self._lang_dir(language), "zscore_results.csv")
        self._write(df, path)
