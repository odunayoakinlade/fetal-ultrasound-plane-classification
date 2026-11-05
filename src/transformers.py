"""Image preprocessing functions and scikit-learn transformers.

The functions do the work on a single image; the classes wrap them so they can be
chained in a scikit-learn Pipeline (see pipeline_builder.py). Only SiteIQRFlagger
learns anything in fit, and it is fitted on the development set only.
"""

import hashlib

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.fft import dctn
from scipy.stats import spearmanr
from skimage.morphology import convex_hull_image
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Hashing (duplicate detection)
# ---------------------------------------------------------------------------

def md5_hash(path):
    """Fingerprint of the file's bytes: identical files share the same MD5."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def perceptual_hash(image):
    """64-bit hash of what an image looks like; similar images get similar bits."""
    small = np.asarray(image.convert("L").resize((32, 32), Image.LANCZOS), dtype=float)
    low = dctn(small, norm="ortho")[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def image_phash(path):
    with Image.open(path) as image:
        return perceptual_hash(image)


# ---------------------------------------------------------------------------
# Fan mask and measurements
# ---------------------------------------------------------------------------

def load_grayscale(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def fan_mask(gray):
    """True inside the ultrasound fan, False outside it."""
    mask = ndimage.binary_opening(gray > 8, iterations=2)
    labels, n = ndimage.label(mask)
    largest = np.argmax(ndimage.sum(mask, labels, range(1, n + 1))) + 1
    return convex_hull_image(labels == largest)


def fan_brightness(path):
    gray = load_grayscale(path)
    return gray[fan_mask(gray)].mean()


def measure(path):
    """Eight summary measurements of one image."""
    gray = load_grayscale(path)
    mask = fan_mask(gray)
    inside = gray[mask]
    counts = np.bincount(inside, minlength=256)
    p = counts[counts > 0] / counts.sum()
    return {
        "brightness": inside.mean(),
        "contrast": inside.std(),
        "entropy": -(p * np.log2(p)).sum(),
        "sharpness": ndimage.laplace(gray.astype(float))[mask].var(),
        "fan_area": mask.mean(),
        "width": gray.shape[1],
        "height": gray.shape[0],
        "aspect_ratio": gray.shape[1] / gray.shape[0],
    }


# ---------------------------------------------------------------------------
# Standardisation steps
# ---------------------------------------------------------------------------

def crop_to_fan(gray, mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def trim_extremes(gray, mask, low=1, high=99):
    low_value, high_value = np.percentile(gray[mask], [low, high])
    trimmed = np.clip(gray, low_value, high_value)
    trimmed[~mask] = 0
    return trimmed.astype(np.uint8)


def pad_and_resize(gray, size=224):
    height, width = gray.shape
    side = max(height, width)
    square = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    square[top:top + height, left:left + width] = gray
    return np.asarray(Image.fromarray(square).resize((size, size), Image.BILINEAR))


def standardise_steps(path):
    """Every intermediate image, used to draw Figure 4.1."""
    gray = load_grayscale(path)
    mask = fan_mask(gray)
    masked = np.where(mask, gray, 0).astype(np.uint8)
    cropped, cropped_mask = crop_to_fan(masked, mask)
    trimmed = trim_extremes(cropped, cropped_mask)
    return {"original": gray, "masked": masked, "cropped": cropped, "trimmed": trimmed,
            "resized": pad_and_resize(trimmed)}


def to_model_input(image):
    """Scale to [0, 1], copy into three channels and apply the ImageNet constants."""
    scaled = image / 255.0
    three_channels = np.stack([scaled, scaled, scaled])
    return (three_channels - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]


def augment(image, rng):
    """One random training variant of a standardised image."""
    x = image / 255.0
    x = x * rng.uniform(0.8, 1.2)
    x = (x - x.mean()) * rng.uniform(0.8, 1.2) + x.mean()
    x = np.clip(x, 0, 1) ** rng.uniform(0.8, 1.2)
    x = ndimage.gaussian_filter(x, sigma=rng.uniform(0, 1.5))
    x = x * (1 + rng.normal(0, 0.1, x.shape))
    x = ndimage.rotate(x, rng.uniform(-10, 10), reshape=False, order=1)
    if rng.random() < 0.5:
        x = x[:, ::-1]
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Collinearity
# ---------------------------------------------------------------------------

def vif_table(df):
    centred = (df - df.mean()).to_numpy()
    return pd.Series([variance_inflation_factor(centred, i) for i in range(df.shape[1])], index=df.columns)


def drop_collinear(measurements, mi, max_rho=0.85, max_vif=10):
    """Drop the weaker of each highly correlated pair, then drop until every VIF <= max_vif.

    "Weaker" means lower mutual information with the plane. Returns the kept columns and
    the reason each other column was dropped.
    """
    rho = pd.DataFrame(spearmanr(measurements).statistic, index=measurements.columns,
                       columns=measurements.columns)
    kept, dropped = list(measurements.columns), {}

    for i, a in enumerate(measurements.columns):
        for b in measurements.columns[i + 1:]:
            if a in kept and b in kept and abs(rho.loc[a, b]) >= max_rho:
                weaker, stronger = (a, b) if mi[a] < mi[b] else (b, a)
                kept.remove(weaker)
                dropped[weaker] = f"|rho| = {abs(rho.loc[a, b]):.2f} with {stronger}"

    while True:
        vif = vif_table(measurements[kept])
        too_high = vif[vif > max_vif]
        if too_high.empty:
            return kept, dropped
        weakest = mi[too_high.index].idxmin()
        kept.remove(weakest)
        dropped[weakest] = f"VIF = {vif[weakest]:.1f}"


# ---------------------------------------------------------------------------
# Fitted transformer: outlier flagging
# ---------------------------------------------------------------------------

class SiteIQRFlagger(BaseEstimator):
    """Learns 1.5 x IQR bounds per site in fit, and flags values outside them."""

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, values, sites):
        grouped = pd.Series(values).groupby(pd.Series(sites).to_numpy())
        q1, q3 = grouped.quantile(0.25), grouped.quantile(0.75)
        iqr = q3 - q1
        self.bounds_ = pd.DataFrame({"Q1": q1, "Q3": q3, "IQR": iqr,
                                     "lower bound": q1 - self.k * iqr, "upper bound": q3 + self.k * iqr})
        return self

    def flag(self, values, sites):
        values = pd.Series(values)
        sites = pd.Series(sites, index=values.index)
        lower = sites.map(self.bounds_["lower bound"])
        upper = sites.map(self.bounds_["upper bound"])
        return (values < lower) | (values > upper)


# ---------------------------------------------------------------------------
# Image pipeline steps (nothing is learned in fit)
# ---------------------------------------------------------------------------

class _ImageStep(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True


class GrayscaleLoader(_ImageStep):
    """File paths -> grayscale images."""

    def transform(self, paths):
        return [load_grayscale(path) for path in paths]


class FanMasker(_ImageStep):
    """Images -> (masked image, mask) pairs, with everything outside the fan set to black."""

    def transform(self, images):
        pairs = []
        for gray in images:
            mask = fan_mask(gray)
            pairs.append((np.where(mask, gray, 0).astype(np.uint8), mask))
        return pairs


class FanCropper(_ImageStep):
    """Crops each (image, mask) pair to the fan's bounding box."""

    def transform(self, pairs):
        return [crop_to_fan(image, mask) for image, mask in pairs]


class ExtremeTrimmer(_ImageStep):
    """Clips pixels inside the fan to each image's own percentiles."""

    def __init__(self, low=1, high=99):
        self.low = low
        self.high = high

    def transform(self, pairs):
        return [trim_extremes(image, mask, self.low, self.high) for image, mask in pairs]


class PadResizer(_ImageStep):
    """Pads to a square and resizes, so no anatomy is stretched."""

    def __init__(self, size=224):
        self.size = size

    def transform(self, images):
        return np.stack([pad_and_resize(image, self.size) for image in images])


class ModelInputScaler(_ImageStep):
    """224 x 224 images -> 3 x 224 x 224 arrays scaled for the pretrained backbone."""

    def transform(self, images):
        return np.stack([to_model_input(image) for image in images]).astype(np.float32)
