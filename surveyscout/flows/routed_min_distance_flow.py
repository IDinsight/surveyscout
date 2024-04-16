from surveyscout.tasks.compute_cost import (
    get_enum_target_osrm_matrix,
)
from surveyscout.tasks.models import (
    min_target_optimization_model,
    recursive_min_target_optimization,
)
from surveyscout.tasks.postprocessing import postprocess_results


def basic_routed_min_distance_flow(
    enum_locations,
    target_locations,
    min_target,
    max_target,
    max_distance,
    max_total_distance,
    routing="osrm",
):
    """
    Executes a basic flow for mapping enumerators to targets with the objective of minimizing
    the total distance traveled, while respecting the given targets and distance constraints.

    This function computes a distance matrix between enumerators and target locations using a
    specified routing API, applies an optimization model to find the minimum cost assignment,
    and then post-processesthe optimized assignment matrix to generate actionable results.

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

    routing : str, optional
        The routing API to use for calculating the distance matrix. Can be either "osrm" or "google".
        Defaults to "osrm".

    Returns
    -------
    results : pd.DataFrame
        A Dataframe containing the post-processed results of the target assignments.



    """

    if routing == "osrm":
        matrix_df = get_enum_target_osrm_matrix(
            enum_locations=enum_locations, target_locations=target_locations
        )
    else:
        raise ValueError(f"Routing API {routing} is not supported yet.")

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


def recursive_routed_min_distance_flow(
    enum_locations,
    target_locations,
    min_target,
    max_target,
    max_distance,
    max_total_distance,
    param_increment=5,
    routing="osrm",
):
    """
    Implements the recursive min target optimization model.

    This function first calculates the distance matrix between enumerators and targets using a specified API.
    Then,it recursively applies optimization to this matrix until an acceptable solution is found or the
    constraints are fully relaxed. After finding a solution, it post-processes the results to provide
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

    param_increment : int, optional
       The percentage increment used to adjust parameter values during the optimization
       recursion if a solution cannot be found. Defaults to 5.
    routing : str, optional
        The routing API to use for calculating the distance matrix. Can be either "osrm" or "google".
        Defaults to "osrm".
    Returns
    -------
    tuple
        A tuple containing two elements: the first being the post-processed results of the target assignments,
        and the second a dictionary of the parameters that led to a solution.
    ```
    """
    if routing == "osrm":
        matrix_df = get_enum_target_osrm_matrix(
            enum_locations=enum_locations, target_locations=target_locations
        )
    else:
        raise ValueError(f"Routing API {routing} is not supported yet.")

    min_possible_max_distance = matrix_df.min(axis=1).max()

    if max_distance <= min_possible_max_distance:
        max_distance = min_possible_max_distance

    results_matrix, params = recursive_min_target_optimization(
        cost_matrix=matrix_df.values,
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
