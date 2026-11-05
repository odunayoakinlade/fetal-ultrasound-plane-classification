"""Pipeline and ColumnTransformer definitions.

Anything that learns from data is fitted on the development (or training) partition only,
so no statistics from validation or test data leak into preprocessing.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.transformers import (CollinearityDropper, ExtremeTrimmer, FanCropper, FanMasker,
                              GrayscaleLoader, ModelInputScaler, PadResizer)


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


def build_feature_transformer(columns, max_rho=0.85, max_vif=10, random_state=None):
    """Measurement columns -> collinearity screening -> standardisation, in one ColumnTransformer.

    Fit on the development set only: the dropper learns which columns to keep and the scaler
    learns their mean and standard deviation.
    """
    measurement_pipeline = Pipeline([
        ("drop_collinear", CollinearityDropper(max_rho=max_rho, max_vif=max_vif, random_state=random_state)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([("measurements", measurement_pipeline, columns)], remainder="drop")


def build_embedding_pca():
    """Standardises embeddings and fits PCA; fit on the development set only."""
    return Pipeline([("scale", StandardScaler()), ("pca", PCA())])


def transform_in_batches(pipeline, items, batch_size=200):
    """Runs a fit-free pipeline over many images without holding all originals in memory."""
    items = list(items)
    return np.concatenate([pipeline.transform(items[start:start + batch_size])
                           for start in range(0, len(items), batch_size)])
