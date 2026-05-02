"""Custom exception hierarchy for the conformal prediction pipeline."""


class ConformalBaseError(Exception):
    """Base exception for all project errors."""


class DataLoadError(ConformalBaseError):
    """Raised when the dataset cannot be loaded from disk."""


class ChecksumError(ConformalBaseError):
    """Raised when a file SHA-256 checksum does not match the stored sidecar value."""


class ValidationError(ConformalBaseError):
    """Raised when pandera schema validation fails on input data."""


class PredictionError(ConformalBaseError):
    """Raised when the prediction pipeline fails; mapped to FastAPI HTTP 422."""


class SkewError(ConformalBaseError):
    """Raised when training-serving feature drift exceeds the PSI threshold."""


class ModelNotFoundError(ConformalBaseError):
    """Raised when a serialized model artefact is missing on disk."""
