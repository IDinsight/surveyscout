from optimization.tasks.preprocessing import (
    get_enum_target_haversine_matrix,
    get_enum_target_matrix_api,
)
from optimization.tasks.models import (
    min_target_optimization_model,
    recursive_min_target_optimization,
)
from optimization.tasks.postprocessing import postprocess_results


def basic_min_osrm_distance_flow(
    enum_locations,
    target_locations,
    min_target,
    max_target,
    max_distance,
    max_total_distance,
):
    """
    Executes a basic flow for mapping enumerators to targets with the objective of minimizing
    the total distance traveled, while respecting the given targets and distance constraints.

    This function computes a Haversine distance matrix between enumerators and target locations,
    applies an optimization model to find the minimum cost assignment, and then post-processes
    the optimized assignment matrix to generate actionable results.

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

    Returns
    -------
    results : pd.DataFrame
        A Dataframe containing the post-processed results of the target assignments.



    """

    matrix_df = get_enum_target_matrix_api(
        enum_locations=enum_locations, target_locations=target_locations, type="OSRM"
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


def recursive_optimization_flow(
    enum_locations,
    target_locations,
    min_target,
    max_target,
    max_distance,
    max_total_distance,
    param_increment=5,
):
    """
    Implements the recursive min target optimization model.

    This function first calculates the Haversine distance matrix between enumerators and targets. Then,
    it recursively applies optimization to this matrix until an acceptable solution is found or the
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

    Returns
    -------
    tuple
        A tuple containing two elements: the first being the post-processed results of the target assignments,
        and the second a dictionary of the parameters that led to a solution.
    ```
    """
    matrix_df = get_enum_target_haversine_matrix(
        enum_locations=enum_locations,
        target_locations=target_locations,
    )

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
