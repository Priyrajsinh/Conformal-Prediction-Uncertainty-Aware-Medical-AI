"""Patient-friendly NL output for conformal prediction sets (rules C7, C45)."""


def translate(label: int, prediction_set: list[int]) -> str:
    """Convert a prediction set into a plain-English clinical summary.

    An empty or multi-label set signals uncertainty; a singleton set is
    reported as high-confidence in the direction of *label*.
    """
    if len(prediction_set) != 1:
        return "Uncertain — recommend follow-up review by a clinician."
    return (
        "Likely heart disease (high confidence)."
        if label == 1
        else "Likely no heart disease (high confidence)."
    )
