"""
Test assignment algorithms.
"""

import pytest

from surveyscout.tasks.models import (
    min_target_optimization_model,
    recursive_min_target_optimization,
)
from surveyscout.tasks.preprocessing import get_enum_target_haversine_matrix


params = [
    [0, 10, 42, 500],
    [2, 4, 35, 300],
]

functions = [min_target_optimization_model, recursive_min_target_optimization]

parametrize_args = [(p, f) for p in params for f in functions]


@pytest.fixture(scope="module")
def enum_target_matrix(enum_df, target_df):
    """Load test target-enumerator cost matrix."""
    return get_enum_target_haversine_matrix(enum_df, target_df).values


@pytest.fixture(scope="module")
def param(request):
    return request.param


@pytest.mark.parametrize("param, function", parametrize_args)
def test_each_target_has_only_one_enum(param, function, enum_target_matrix):
    min_target, max_target, max_cost, max_total_cost = param
    assignment_matrix = function(
        enum_target_matrix, min_target, max_target, max_cost, max_total_cost
    )
    if function == recursive_min_target_optimization:
        assignment_matrix = assignment_matrix[0]
    assert (assignment_matrix.sum(axis=1) == 1.0).all()


@pytest.mark.parametrize("param, function", parametrize_args)
def test_target_constraints_are_met(param, function, enum_target_matrix):
    min_target, max_target, max_cost, max_total_cost = param
    assignment_matrix = function(
        enum_target_matrix, min_target, max_target, max_cost, max_total_cost
    )
    if function == recursive_min_target_optimization:
        assignment_matrix = assignment_matrix[0]
    assert (assignment_matrix.sum(axis=0) >= min_target).all()
    assert (assignment_matrix.sum(axis=0) <= max_target).all()


@pytest.mark.parametrize("param, function", parametrize_args)
def test_cost_constraints_are_met(param, function, enum_target_matrix):
    min_target, max_target, max_cost, max_total_cost = param
    assignment_matrix = function(
        enum_target_matrix, min_target, max_target, max_cost, max_total_cost
    )
    if function == recursive_min_target_optimization:
        assignment_matrix = assignment_matrix[0]
    assigned_distance_df = assignment_matrix * enum_target_matrix
    assert (assigned_distance_df <= max_cost).all().all()
    assert (assigned_distance_df.sum(axis=0) <= max_total_cost).all()
