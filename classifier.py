"""The group's pipeline, called by the application (app.py).

This is the only file of the application to be changed. Implement two functions:

    load_model()            -> called once, when the application starts
    predict(model, frames)  -> called for every recording

`frames` is a pandas DataFrame in the format of the CSV files of the dataset, one row
per frame at 30 frames per second, with the columns

    frame, timestamp, confidence, x_0 ... x_67, y_0 ... y_67

in pixels (origin at the top-left corner, y increasing downwards). The coordinates are
NOT normalised: apply the same normalisation and compute the features with the same
code that was used for training, imported from your own modules, not rewritten here.

`predict` returns a dict with one probability per emotion, in the order of EMOTIONS.
The placeholder below returns the same probability for every emotion.
"""
import numpy as np
import pandas as pd

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def load_model():
    """Load and return the fitted pipeline, for example with joblib.load("model.joblib")."""
    return None


def predict(model, frames: pd.DataFrame) -> dict:
    """Return {emotion: probability} for one recording."""
    # 1. normalise the coordinates and compute the features of this recording (one row)
    # 2. probabilities = model.predict_proba(features)[0]
    # 3. return dict(zip(EMOTIONS, probabilities)), in the order of model.classes_
    probabilities = np.full(len(EMOTIONS), 1.0 / len(EMOTIONS))
    return dict(zip(EMOTIONS, probabilities))
