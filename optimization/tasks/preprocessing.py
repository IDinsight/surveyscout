import numpy as np
import pandas as pd
import requests
from optimization.config import OSRM_URL
from optimization.utils import format_url_with_coords_osrm, haversine


def get_enum_target_haversine_matrix(
    enum_locations,
    target_locations,
    *args,
    **kwargs,
):
    """
    Get the haversine distance matrix between enumerators and targets.

    Parameters
    ----------
    enum_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of enumerators.

    target_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of targets, with a similar structure to `enum_locations`.


    Returns
    -------
    pd.DataFrame
        Haversine distance matrix between enumerators and targets.
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


def get_enum_target_matrix_api(enum_locations, target_locations, type="OSRM"):
    enums_coords = enum_locations.get_gps_coords()
    targets_coords = target_locations.get_gps_coords()
    if type == "OSRM":
        url = OSRM_URL + "/table/v1/driving/"
        matrix = _get_enum_target_matrix_osrm(url, targets_coords, enums_coords)
    else:
        raise ValueError(f"Unknown type {type}")
    return matrix


def _get_enum_target_matrix_osrm(url, target_coords, enum_coords):
    """Get the matrix of distances between enumerators and targets
    using OSRM."""
    matrix = [
        _get_unique_target_enum_row_osrm(url, target_coord, enum_coords)
        for target_coord in target_coords
    ]

    return matrix


def _get_unique_target_enum_row_osrm(url, target_coord, enum_coords):
    """Get the row of the matrix for a target coordinate."""
    coords = [target_coord] + enum_coords
    url = format_url_with_coords_osrm(url, coords)
    response = requests.get(url)
    data = response.json()
    return data
