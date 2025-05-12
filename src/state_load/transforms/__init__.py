"""
This module is used to compute in-memory transforms, e.g. PCA, which would cause leakage
if computed at the whole dataset level. This is not an issue with, say, UCE, which is a
foundation model trained on other data.
"""

from .base import BaseTransform
from .pca import PCATransform
