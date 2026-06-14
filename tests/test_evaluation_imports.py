"""Smoke tests for offline evaluation script imports (WP-14)."""

import importlib


def test_run_comprehensive_test_imports():
    importlib.import_module("src.pipeline.evaluation.run_comprehensive_test")


def test_generate_xai_report_imports():
    importlib.import_module("src.pipeline.xai.generate_xai_report")


def test_generate_validation_predictions_imports():
    importlib.import_module("src.pipeline.inference.generate_validation_predictions")
