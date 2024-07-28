import logging

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


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def basic_min_distance_flow(
    enum_locations: LocationDataset,
    target_locations: LocationDataset,
    min_target: int,
    max_target: int,
    max_distance: float,
    max_total_distance: float,
    cost_function="haversine",
) -> pd.DataFrame:
    """
    Executes a basic flow for mapping enumerators to targets with the objective of
    minimizing the total distance traveled, while respecting the given targets and
    distance constraints.

    This function computes a distance matrix between enumerators and target
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

    cost_function : str
        The cost function to use for calculating the distance matrix.
        Can be "haversine", "osrm".
        Defaults to "haversine".
    Returns
    -------
    results : pd.DataFrame
        A Dataframe containing the post-processed results of the target assignments.



    """
    if cost_function == "haversine":
        cost_func = get_enum_target_haversine_matrix
    elif cost_function == "osrm":
        cost_func = get_enum_target_osrm_matrix
    else:
        raise ValueError(
            "Invalid routing method. Please choose from 'haversine' or 'osrm'."
        )
    logger.info(f"Calculating distance matrix using {cost_function} routing method.")
    logger.info(
        f"Number of enumerators: {len(enum_locations.get_ids())}, Number of targets: {len(target_locations.get_ids())}"
        f"Total number of possible assignments: {len(enum_locations.get_ids()) * len(target_locations.get_ids())}"
    )
    matrix_df = cost_func(
        enum_locations=enum_locations, target_locations=target_locations
    )
    logger.info("Distance matrix calculated successfully.")

    logger.info("Applying optimization model to find minimum cost assignment.")
    results_matrix = min_target_optimization_model(
        cost_matrix=matrix_df.values,
        min_target=min_target,
        max_target=max_target,
        max_cost=max_distance,
        max_total_cost=max_total_distance,
    )
    logger.info(f"Successfully assigned {len(results_matrix)} targets to enumerators.")
    results = postprocess_results(
        results=results_matrix,
        enum_locations=enum_locations,
        target_locations=target_locations,
    )
    logger.info(f"Total cost: {results['cost'].sum()}")
    return results


def recursive_min_distance_flow(
    enum_locations: LocationDataset,
    target_locations: LocationDataset,
    min_target: int,
    max_target: int,
    max_distance: float,
    max_total_distance: float,
    cost_function="haversine",
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

    cost_function : str
        The cost function method to use for calculating the distance matrix.
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
    if cost_function == "haversine":
        cost_func = get_enum_target_haversine_matrix
    elif cost_function == "osrm":
        cost_func = get_enum_target_osrm_matrix
    else:
        raise ValueError(
            "Invalid routing method. Please choose from 'haversine' or 'osrm'."
        )

    logger.info(f"Calculating distance matrix using {cost_function} routing method.")
    logger.info(
        f"Number of enumerators: {len(enum_locations.get_ids())}, Number of targets: {len(target_locations.get_ids())}"
        f"Total number of possible assignments: {len(enum_locations.get_ids()) * len(target_locations.get_ids())}"
    )

    matrix_df = cost_func(
        enum_locations=enum_locations, target_locations=target_locations
    )
    min_possible_max_distance = matrix_df.min(axis=1).max()

    if max_distance <= min_possible_max_distance:
        max_distance = min_possible_max_distance
    logger.info("Distance matrix calculated successfully.")

    logger.info("Applying optimization model to find minimum cost assignment.")
    results_matrix, params = recursive_min_target_optimization(
        cost_matrix=matrix_df.values,
        min_target=min_target,
        max_target=max_target,
        max_cost=max_distance,
        max_total_cost=max_total_distance,
        param_increment=param_increment,
    )
    logger.info(f"Successfully assigned {len(results_matrix)} targets to enumerators.")
    results = postprocess_results(
        results=results_matrix,
        enum_locations=enum_locations,
        target_locations=target_locations,
    )
    logger.info(f"Total cost: {results['cost'].sum()}")
    return results, params
