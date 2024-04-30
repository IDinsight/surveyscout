from typing import List, Dict

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from surveyscout.utils import LocationDataset


def get_enum_target_haversine_matrix(
    enum_locations: LocationDataset,
    target_locations: LocationDataset,
    *args: List,
    **kwargs: Dict,
) -> pd.DataFrame:
    """
    Haversine distance matrix between enumerators and targets.


    Parameters
    ----------
    enum_locations : LocationDataset
        A LocationDataset object containing the id and locations of enumerators.

    target_locations : LocationDataset
        A LocationDataset object containing the id and locations of targets, with a similar structure to `enum_locations`.

    Returns
    -------
    pd.DataFrame
        Haversine distance matrix between enumerators and targets.
        Columns are enumerator IDs, rows are target IDs.
    """
    targets_lat = target_locations.get_gps_coords()[:, 0]
    targets_long = target_locations.get_gps_coords()[:, 1]
    enums_lat = enum_locations.get_gps_coords()[:, 0]
    enums_long = enum_locations.get_gps_coords()[:, 1]

    lat1, lat2 = np.meshgrid(targets_lat, enums_lat, indexing="ij")
    lon1, lon2 = np.meshgrid(targets_long, enums_long, indexing="ij")
    matrix = haversine(lat1, lon1, lat2, lon2)
    matrix_df = pd.DataFrame(
        matrix, index=target_locations.get_ids(), columns=enum_locations.get_ids()
    )
    return matrix_df


def haversine(
    lat1: float | NDArray,
    lon1: float | NDArray,
    lat2: float | NDArray,
    lon2: float | NDArray,
) -> float | NDArray:
    """Compute the haversine distance between two GPS coordinates."""
    R = 6371.0

    # convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    return distance
