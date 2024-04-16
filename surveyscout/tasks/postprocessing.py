from typing import List
from numpy.typing import NDArray
import pandas as pd

from surveyscout.utils import LocationDataset


def postprocess_results(
    results: NDArray | List[List],
    enum_locations: LocationDataset,
    target_locations: LocationDataset,
    enum_target_cost_matrix: NDArray | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Processes the raw results from the optimization model, converting them into a
    pandas DataFrame format.
    This function takes the results matrix, which indicates the assignment of
    enumerators to targets with a value of 1, and converts it into a human-readable
    DataFrame. The resulting DataFrame contains only the successful assignments as
    per the optimization model.

    Parameters
    ----------
    results : numpy.ndarray or list of lists
        The raw results matrix from the optimization algorithm, with `1` indicating
        an assigned target and `0` otherwise.

    enum_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of enumerators.

    target_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of targets, with a
        similar structure to `enum_locations`.

    enum_target_cost_matrix : NDArray
        Cost matrix

    **kwargs : dict, optional
        Additional keyword arguments that can be used in a custom postprocessing step.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame where each row represents an assignment with columns corresponding
        to target IDs, enumerator IDs, and assigned values (all 1's).
    """

    if enum_target_cost_matrix is not None:
        results *= enum_target_cost_matrix

    df_results = pd.DataFrame(
        results,
        index=target_locations.get_ids(),
        columns=enum_locations.get_ids(),
    )

    stacked = df_results.stack()
    filtered = stacked[stacked > 0]
    df = filtered.reset_index(level=[0, 1])
    df.columns = [
        "target_id",
        "enum_id",
        "cost",
    ]

    return df
