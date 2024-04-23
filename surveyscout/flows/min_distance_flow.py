from typing import Dict, Union, Tuple
import pandas as pd


from surveyscout.tasks.compute_cost import (
    get_enum_target_haversine_matrix,
    get_enum_target_osrm_matrix,
)
from surveyscout.tasks.models import (
    min_target_optimization_model,
    recursive_min_target_optimization,
)
from surveyscout.tasks.postprocessing import postprocess_results
from surveyscout.utils import LocationDataset


def basic_min_distance_flow(
    enum_locations: LocationDataset,
    target_locations: LocationDataset,
    min_target: int,
    max_target: int,
    max_distance: float,
    max_total_distance: float,
    routing="haversine",
) -> pd.DataFrame:
    """
    Executes a basic flow for mapping enumerators to targets with the objective of
    minimizing the total distance traveled, while respecting the given targets and
    distance constraints.

    This function computes a Haversine distance matrix between enumerators and target
    locations using a specified routing method, applies an optimization model to find
    the minimum cost assignment, and then post-processesthe optimized assignment matrix
    to generate actionable results.

    Parameters
    ----------
    enum_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of enumerators.

    target_locations : class <LocationDataset>
          A <LocationDataset> object containing the id and locations of targets, with a similar structure to `enum_locations`.

    min_target : int
        The minimum number of targets each enumerator is required to visit.

    max_target : int
        The maximum number of targets each enumerator is allowed to visit.

    max_distance : float
        The maximum allowable distance an enumerator can travel to a single target.

    max_total_distance : float
        The maximum total distance each enumerator can travel to visit targets.

    routing : str
        The routing method to use for calculating the distance matrix.
        Can be "haversine", "osrm".
        Defaults to "haversine".
    Returns
    -------
    results : pd.DataFrame
        A Dataframe containing the post-processed results of the target assignments.



    """
    if routing == "haversine":
        matrix_df = get_enum_target_haversine_matrix(
            enum_locations=enum_locations,
            target_locations=target_locations,
        )
    elif routing == "osrm":
        matrix_df = get_enum_target_osrm_matrix(
            enum_locations=enum_locations, target_locations=target_locations
        )
    else:
        raise ValueError(
            "Invalid routing method. Please choose from 'haversine' or 'osrm'."
        )

    results_matrix = min_target_optimization_model(
        cost_matrix=matrix_df.values,
        min_target=min_target,
        max_target=max_target,
        max_cost=max_distance,
        max_total_cost=max_total_distance,
    )

    results = postprocess_results(
        results=results_matrix,
        enum_locations=enum_locations,
        target_locations=target_locations,
    )
    return results


def recursive_min_distance_flow(
    enum_locations: LocationDataset,
    target_locations: LocationDataset,
    min_target: int,
    max_target: int,
    max_distance: float,
    max_total_distance: float,
    routing="haversine",
    param_increment: Union[int, float] = 5,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Implements the recursive min target optimization model.

    This function first calculates the  distance matrix between enumerators
    and targets using aspecified routing API. Then,it recursively applies
    optimization to this matrix until an acceptable solution is found or the
    constraints are fully relaxed. After finding a solution, it post-processes
    the results to provide
    actionable output.

    Parameters
    ----------
    enum_locations : LocationDataset
        A <LocationDataset> object containing the id and locations of enumerators, usually as (latitude, longitude) pairs.

    target_locations : LocationDataset
          A <LocationDataset> object containing the id and locations of targets, with a similar structure to `enum_locations`.

    min_target : int
        The minimum number of targets each enumerator is required to visit.

    max_target : int
        The maximum number of targets each enumerator is allowed to visit.

    max_distance : float
        The maximum total distance each enumerator can travel to visit targets.

    max_total_distance : float
        The maximum total distance each enumerator can travel to visit targets.

    routing : str
        The routing method to use for calculating the distance matrix.
        Can be "haversine", "osrm" or "google".
        Defaults to "haversine".
    param_increment : int
       The percentage increment used to adjust parameter values during the optimization
       recursion if a solution cannot be found. Defaults to 5.

    Returns
    -------
    (pd.DataFrame, Dict)
        A tuple containing two elements: the first being the post-processed results of the target assignments,
        and the second a dictionary of the parameters that led to a solution.
    ```
    """
    if routing == "haversine":
        distance_df = get_enum_target_haversine_matrix(
            enum_locations=enum_locations,
            target_locations=target_locations,
        )
    elif routing == "osrm":
        distance_df = get_enum_target_osrm_matrix(
            enum_locations=enum_locations, target_locations=target_locations
        )
    else:
        raise ValueError(
            "Invalid routing method. Please choose from 'haversine' or 'osrm'."
        )

    min_possible_max_distance = distance_df.min(axis=1).max()

    if max_distance <= min_possible_max_distance:
        max_distance = min_possible_max_distance

    results_matrix, params = recursive_min_target_optimization(
        cost_matrix=distance_df.values,
        min_target=min_target,
        max_target=max_target,
        max_cost=max_distance,
        max_total_cost=max_total_distance,
        param_increment=param_increment,
    )

    results = postprocess_results(
        results=results_matrix,
        enum_locations=enum_locations,
        target_locations=target_locations,
    )
    return results, params
