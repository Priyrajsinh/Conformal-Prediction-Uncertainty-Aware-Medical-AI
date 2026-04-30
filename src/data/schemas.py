"""Pandera schema for the UCI Heart Disease (Cleveland) dataset.

Column ranges reflect the 0-indexed encoding applied during loading
(cp 0-3, slope 0-2, thal 0-2 after remapping raw Cleveland codes).
"""

import pandera as pa
from pandera.pandas import Column, DataFrameSchema

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
