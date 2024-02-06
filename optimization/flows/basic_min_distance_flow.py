from optimization.tasks.preprocessing import get_enum_target_haversine_matrix
from optimization.tasks.models import min_target_optimization_model
from optimization.tasks.postprocessing import postprocess_results


def basic_min_distance_flow(
    enum_locations,
    target_locations,
    min_target,
    max_target,
    max_distance,
    max_total_distance,
):
    """Basic minimum distance flow without recursive parameters update."""
    cost_matrix = get_enum_target_haversine_matrix(
        enum_locations=enum_locations,
        target_locations=target_locations,
    )
    results_matrix = min_target_optimization_model(
        cost_matrix=cost_matrix,
        min_target=min_target,
        max_target=max_target,
        max_distance=max_distance,
        max_total_distance=max_total_distance,
    )
    results = postprocess_results(
        results=results_matrix,
        enum_locations=enum_locations,
        target_locations=target_locations,
    )
    return results
