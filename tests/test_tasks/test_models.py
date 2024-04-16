"""
Test assignment algorithms.
"""

import pytest

from surveyscout.tasks.models import min_target_optimization_model
from surveyscout.tasks.preprocessing import get_enum_target_haversine_matrix


@pytest.fixture(scope="module")
def enum_target_matrix(enum_df, target_df):
    """Load test target-enumerator cost matrix."""
    return get_enum_target_haversine_matrix(enum_df, target_df).values


params = [
    [0, 10, 9000, 100000],
    [2, 4, 10000, 50000],
]


@pytest.fixture(scope="module", params=params, ids=str)
def param(request):
    return request.param


@pytest.fixture(scope="module")
def optimized_assignment_array(enum_target_matrix, param):
    min_target, max_target, max_cost, max_total_cost = param
    return min_target_optimization_model(
        enum_target_matrix, min_target, max_target, max_cost, max_total_cost
    )


def test_each_target_has_only_one_enum(optimized_assignment_array):
    assert (optimized_assignment_array.sum(axis=1) == 1.0).all()


def test_target_constraints_are_met(optimized_assignment_array, param):
    min_target, max_target, _, _ = param
    assert (optimized_assignment_array.sum(axis=0) >= min_target).all()
    assert (optimized_assignment_array.sum(axis=0) <= max_target).all()


def test_cost_constraints_are_met(
    optimized_assignment_array, enum_target_matrix, param
):
    min_target, max_target, max_cost, max_total_cost = param
    assigned_distance_df = optimized_assignment_array * enum_target_matrix
    assert (assigned_distance_df <= max_cost).all().all()
    assert (assigned_distance_df.sum(axis=0) <= max_total_cost).all()
