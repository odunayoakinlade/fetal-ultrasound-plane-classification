"""Pipeline and ColumnTransformer definitions.

Anything that learns from data is fitted on the development (or training) partition only,
so no statistics from validation or test data leak into preprocessing.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.transformers import (ExtremeTrimmer, FanCropper, FanMasker, GrayscaleLoader,
                              ModelInputScaler, PadResizer)


def build_image_pipeline(size=224, include_model_scaling=False):
    """File paths -> standardised size x size images (or 3 x size x size model inputs)."""
    steps = [
        ("load", GrayscaleLoader()),
        ("mask", FanMasker()),
        ("crop", FanCropper()),
        ("trim", ExtremeTrimmer(low=1, high=99)),
        ("resize", PadResizer(size=size)),
    ]
    if include_model_scaling:
        steps.append(("model_input", ModelInputScaler()))
    return Pipeline(steps)


def build_feature_transformer(columns):
    """Keeps the given measurement columns and standardises them; drops the rest."""
    return ColumnTransformer([("scale", StandardScaler(), columns)], remainder="drop")


def transform_in_batches(pipeline, items, batch_size=200):
    """Runs a fit-free pipeline over many images without holding all originals in memory."""
    items = list(items)
    return np.concatenate([pipeline.transform(items[start:start + batch_size])
                           for start in range(0, len(items), batch_size)])
