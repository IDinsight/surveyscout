import numpy as np
import pandas as pd
import requests
from surveyscout.config import OSRM_URL


def get_enum_target_osrm_matrix(enum_locations, target_locations):
    """Get the matrix of distances between enumerators and targets using OSRM api.
    This function calls the OSRM /table/v1/driving/ api endpoint to get the matrix
    of distances between enumerators and targets. The distance represents the
    distance of the fastest route between the two points.
    Parameters
    ----------
    enum_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of enumerators.

    target_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of targets,
         with a similar structure to `enum_locations`.


    Returns
    -------
    pd.DataFrame
        distance matrix between enumerators and targets.
    """
    enums_coords = enum_locations.get_gps_coords()
    targets_coords = target_locations.get_gps_coords()

    url = OSRM_URL + "/table/v1/driving/"
    matrix = _get_enum_target_matrix_osrm(url, targets_coords, enums_coords)
    matrix_df = pd.DataFrame(
        matrix, index=target_locations.get_ids(), columns=enum_locations.get_ids()
    )
    return matrix_df


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
    coords = np.insert(enum_coords, 0, target_coord, axis=0)
    url = _format_url_with_coords_osrm(url, coords)
    response = requests.get(url)
    if "distances" in response.json():
        data = response.json()["distances"][0][1:]
        return np.array(data) / 1000

    else:
        return []


def _format_url_with_coords_osrm(url, coords):
    """Formats URL with GPS coordinates for OSRM API"""
    coord_str = ";".join([f"{x[1]},{x[0]}" for x in coords])
    url = url + coord_str + "?sources=0&annotations=distance,duration"
    return url
