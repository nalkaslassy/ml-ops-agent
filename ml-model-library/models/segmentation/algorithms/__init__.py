"""
Segmentation algorithm registry.
"""

from . import kmeans, dbscan

ALGORITHM_REGISTRY = [
    kmeans,
    dbscan,
]
