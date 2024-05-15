from typing import Dict, Union, Tuple
import pandas as pd


from surveyscout.tasks.compute_cost import (
    get_enum_target_haversine_matrix,
    get_enum_target_osrm_matrix,
    get_enum_target_google_distance_matrix,
)
from surveyscout.tasks.models import (
    min_target_optimization_model,
    recursive_min_target_optimization,
)
from surveyscout.tasks.postprocessing import postprocess_results
from surveyscout.utils import LocationDataset


def get_cost_matrix(
    enum_locations: LocationDataset,
    target_locations: LocationDataset,
    cost_function: str,
) -> pd.DataFrame:
    match cost_function:
        case "haversine":
            return get_enum_target_haversine_matrix(
                enum_locations=enum_locations, target_locations=target_locations
            )
        case "osrm":
            return get_enum_target_osrm_matrix(
                enum_locations=enum_locations, target_locations=target_locations
            )
        case "google_duration":
            return get_enum_target_google_distance_matrix(
                enum_locations=enum_locations,
                target_locations=target_locations,
                value="duration",
            )
        case "google_distance":
            return get_enum_target_google_distance_matrix(
                enum_locations=enum_locations,
                target_locations=target_locations,
                value="distance",
            )
        case _:
            raise ValueError(
                "Invalide routing method. Please choose from "
                "'haversine', 'osrm', 'google_distance', or 'google_duration'."
            )


def basic_min_distance_flow(
    enum_locations: LocationDataset,
    target_locations: LocationDataset,
    min_target: int,
    max_target: int,
    max_cost: float,
    max_total_cost: float,
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

    max_cost : float
        The maximum allowable cost an enumerator can travel to a single target.

    max_total_cost : float
        The maximum total cost each enumerator can travel to visit targets.

    cost_function : str
        The cost function to use for calculating the distance matrix.
        Can be "haversine", "osrm", "google_distance", or "google_duration".
        Note that for "google_duration" the unit cost is 1 second, and for all others
        the unit cost is 1 meter.
        Defaults to "haversine".

    Returns
    -------
    results : pd.DataFrame
        A Dataframe containing the post-processed results of the target assignments.
    """
    cost_matrix = get_cost_matrix(
        enum_locations=enum_locations,
        target_locations=target_locations,
        cost_function=cost_function,
    )

    min_possible_max_distance = cost_matrix.min(axis=1).max()

    if max_cost < min_possible_max_distance:
        raise ValueError(
            f"Minimum possible `max_distance` is {min_possible_max_distance}. "
            "Please provide a value greater than or equal to this."
        )

    results_matrix = min_target_optimization_model(
        cost_matrix=cost_matrix.values,
        min_target=min_target,
        max_target=max_target,
        max_cost=max_cost,
        max_total_cost=max_total_cost,
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
    max_cost: float,
    max_total_cost: float,
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

    max_cost : float
        The maximum allowable cost an enumerator can travel to a single target.

    max_total_cost : float
        The maximum total cost each enumerator can travel to visit targets.

    cost_function : str
        The cost function to use for calculating the distance matrix.
        Can be "haversine", "osrm", "google_distance", or "google_duration".
        Note that for "google_duration" the unit cost is 1 second, and for all others
        the unit cost is 1 meter.
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
    cost_matrix = get_cost_matrix(
        enum_locations=enum_locations,
        target_locations=target_locations,
        cost_function=cost_function,
    )

    min_possible_max_distance = cost_matrix.min(axis=1).max()

    if max_cost <= min_possible_max_distance:
        max_cost = min_possible_max_distance

    results_matrix, params = recursive_min_target_optimization(
        cost_matrix=cost_matrix.values,
        min_target=min_target,
        max_target=max_target,
        max_cost=max_cost,
        max_total_cost=max_total_cost,
        param_increment=param_increment,
    )

    results = postprocess_results(
        results=results_matrix,
        enum_locations=enum_locations,
        target_locations=target_locations,
    )
    return results, params
