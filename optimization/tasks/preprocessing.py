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
    targets = target_locations.get_gps_coords()
    enums = enum_locations.get_gps_coords()

    matrix = np.zeros((len(targets), len(enums)))

    for i in range(len(targets)):
        for j in range(len(enums)):
            matrix[i, j] = haversine(
                targets[i][1], targets[i][0], enums[j][1], enums[j][0]
            )
    matrix_df = pd.DataFrame(
        matrix, index=target_locations.get_ids(), columns=enum_locations.get_ids()
    )
    return matrix_df
