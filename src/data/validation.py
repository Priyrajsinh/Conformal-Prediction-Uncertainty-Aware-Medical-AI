"""Pandera-based DataFrame validation for the heart disease pipeline."""

import pandas as pd
import pandera as pa

from src.data.schemas import HEART_DISEASE_SCHEMA
from src.exceptions import ValidationError
from src.logger import get_logger

logger = get_logger(__name__)


def validate_heart_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate *df* against HEART_DISEASE_SCHEMA; raise ValidationError on failure."""
    try:
        validated = HEART_DISEASE_SCHEMA.validate(df)
        logger.info("Schema validation passed: %d rows", len(validated))
        return validated
    except pa.errors.SchemaError as exc:
        raise ValidationError(str(exc)) from exc
