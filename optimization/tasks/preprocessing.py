from optimization.utils import get_intergroup_haversine_matrix


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
        class containing locations informations
    target_locations : class <LocationDataset>
        class containing locations informations

    Returns
    -------
    pd.DataFrame
        Haversine distance matrix between enumerators and targets.
    """
    return get_intergroup_haversine_matrix(
        row_locations=enum_locations,
        col_locations=target_locations,
    )
