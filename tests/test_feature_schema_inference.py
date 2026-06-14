"""Feature schema validation at inference (WP-04)."""

import numpy as np
import pytest

from src.utils.model_loader import validate_feature_schema


def test_validate_feature_schema_match():
    schema = {"feature_count": 64}
    assert validate_feature_schema((1, 64), schema) is True


def test_validate_feature_schema_mismatch_raises():
    schema = {"feature_count": 64}
    with pytest.raises(ValueError, match="Feature count mismatch"):
        validate_feature_schema((1, 32), schema)
