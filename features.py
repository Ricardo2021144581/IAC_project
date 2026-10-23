
"""Feature extraction matching the project's Question 3 notebook."""

import numpy as np
import pandas as pd


MEASURES = [f"M{i}" for i in range(1, 12)]
STATISTICS = ["mean", "std", "range", "v", "E"]
FEATURE_NAMES = [
    f"{measure}_{stat}"
    for measure in MEASURES
    for stat in STATISTICS
]


def distance(df, a, b):
    """Euclidean distance between two facial landmarks."""
    return np.sqrt(
        (df[f"x_{a}"] - df[f"x_{b}"]) ** 2
        + (df[f"y_{a}"] - df[f"y_{b}"]) ** 2
    )


def extract_video_features(frames: pd.DataFrame) -> pd.DataFrame:
    """Convert one recording's raw pixel coordinates into 55 features."""

    if frames.empty:
        raise ValueError("The recording contains no frames.")

    frames = frames.sort_values("frame").reset_index(drop=True)

    # Normalise coordinates: origin at landmark 27, scale by
    # the distance between landmarks 36 and 45.
    d = np.sqrt(
        (frames["x_45"] - frames["x_36"]) ** 2
        + (frames["y_45"] - frames["y_36"]) ** 2
    )

    if (d <= 0).any() or not np.isfinite(d).all():
        raise ValueError("Invalid interocular distance in the recording.")

    norm = pd.DataFrame({"frame": frames["frame"]})

    for k in range(68):
        norm[f"x_{k}"] = (frames[f"x_{k}"] - frames["x_27"]) / d
        norm[f"y_{k}"] = (frames[f"y_{k}"] - frames["y_27"]) / d

    # Original six measures from Question 3.1.
    m = pd.DataFrame({"frame": norm["frame"]})

    m["M1"] = distance(norm, 66, 62)
    m["M2"] = distance(norm, 54, 48)

    m["M3"] = (
        (norm["y_51"] + norm["y_57"]) / 2
        - (norm["y_48"] + norm["y_54"]) / 2
    )

    m["M4"] = (
        (
            (norm["y_36"] + norm["y_37"] + norm["y_38"]
             + norm["y_39"] + norm["y_40"] + norm["y_41"]) / 6
            - (norm["y_17"] + norm["y_18"] + norm["y_19"]
               + norm["y_20"] + norm["y_21"]) / 5
        )
        + (
            (norm["y_42"] + norm["y_43"] + norm["y_44"]
             + norm["y_45"] + norm["y_46"] + norm["y_47"]) / 6
            - (norm["y_22"] + norm["y_23"] + norm["y_24"]
               + norm["y_25"] + norm["y_26"]) / 5
        )
    ) / 2

    m["M5"] = (
        distance(norm, 37, 41)
        + distance(norm, 38, 40)
        + distance(norm, 43, 47)
        + distance(norm, 44, 46)
    ) / 4

    m["M6"] = distance(norm, 21, 22)

    # Additional five measures from Question 3.2.
    m["M7"] = distance(norm, 33, 51)
    m["M8"] = distance(norm, 31, 35)

    m["M9"] = (
        (norm["y_19"] + norm["y_24"]) / 2
        - (norm["y_21"] + norm["y_22"]) / 2
    )

    m["M10"] = distance(norm, 57, 8)

    m["M11"] = (
        (norm["y_36"] + norm["y_39"]) / 2
        - (norm["y_42"] + norm["y_45"]) / 2
    )

    # Five statistics per measure, matching the notebook:
    # pandas std() uses ddof=1; v and E are computed per frame.
    result = {}
    n = len(m)

    for name in MEASURES:
        values = m[name]
        differences = values.diff()

        result[f"{name}_mean"] = values.mean()
        result[f"{name}_std"] = values.std()
        result[f"{name}_range"] = values.max() - values.min()

        if n > 1:
            result[f"{name}_v"] = differences.abs().sum() / (n - 1)
            result[f"{name}_E"] = differences.pow(2).sum() / (n - 1)
        else:
            result[f"{name}_v"] = 0.0
            result[f"{name}_E"] = 0.0

    # Explicit order is important: it must match model training.
    return pd.DataFrame(
        [[result[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )
