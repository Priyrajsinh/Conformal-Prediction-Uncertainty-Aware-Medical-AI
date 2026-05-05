"""Pandera schema and Pydantic I/O models for the UCI Heart Disease dataset.

Column ranges reflect the 0-indexed encoding applied during loading
(cp 0-3, slope 0-2, thal 0-2 after remapping raw Cleveland codes).
"""

import pandera as pa
from pandera.pandas import Column, DataFrameSchema
from pydantic import BaseModel, Field

HEART_DISEASE_SCHEMA = DataFrameSchema(
    {
        "age": Column(int, checks=[pa.Check.ge(29), pa.Check.le(77)]),
        "sex": Column(int, checks=pa.Check.isin([0, 1])),
        "cp": Column(int, checks=[pa.Check.ge(0), pa.Check.le(3)]),
        "trestbps": Column(int, checks=[pa.Check.ge(94), pa.Check.le(200)]),
        "chol": Column(int, checks=[pa.Check.ge(126), pa.Check.le(564)]),
        "fbs": Column(int, checks=pa.Check.isin([0, 1])),
        "restecg": Column(int, checks=[pa.Check.ge(0), pa.Check.le(2)]),
        "thalach": Column(int, checks=[pa.Check.ge(71), pa.Check.le(202)]),
        "exang": Column(int, checks=pa.Check.isin([0, 1])),
        "oldpeak": Column(float, checks=[pa.Check.ge(0.0), pa.Check.le(6.2)]),
        "slope": Column(int, checks=[pa.Check.ge(0), pa.Check.le(2)]),
        "ca": Column(int, checks=[pa.Check.ge(0), pa.Check.le(4)]),
        "thal": Column(int, checks=[pa.Check.ge(0), pa.Check.le(3)]),
        "target": Column(int, checks=pa.Check.isin([0, 1])),
    }
)


class HeartFeatures(BaseModel):
    """13 UCI heart disease features with validated ranges for the FastAPI endpoint."""

    age: int = Field(ge=29, le=77)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: int = Field(ge=94, le=200)
    chol: int = Field(ge=126, le=564)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=71, le=202)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0.0, le=6.2)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)


class ConformalOutput(BaseModel):
    """Conformal prediction response returned by POST /api/v1/predict."""

    label: int
    prediction_set: list[int]
    coverage_guarantee: float
    mean_set_size: float
    alpha_used: float
    nl_summary: str
