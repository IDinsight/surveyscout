import numpy as np
import pandas as pd
from optimization.utils import haversine


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
