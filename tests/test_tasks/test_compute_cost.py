import numpy as np
from numpy.typing import NDArray
import pytest
from surveyscout.tasks.compute_cost.haversine import get_enum_target_haversine_matrix
from surveyscout.tasks.compute_cost.osrm import get_enum_target_osrm_matrix
from surveyscout.utils import LocationDataset

"""
Test cost computation functions.
"""


@pytest.fixture(scope="module")
def enum_target_haversine_matrix(
    enum_df: LocationDataset, target_df: LocationDataset
) -> NDArray:
    return get_enum_target_haversine_matrix(
        enum_locations=enum_df, target_locations=target_df
    )


@pytest.fixture(scope="module")
def enum_target_osrm_matrix(
    enum_df: LocationDataset, target_df: LocationDataset
) -> NDArray:
    return get_enum_target_osrm_matrix(
        enum_locations=enum_df, target_locations=target_df
    )


@pytest.fixture(params=["osrm", "haversine"])
def cost_matrix(
    request: pytest.FixtureRequest,
    enum_df: LocationDataset,
    target_df: LocationDataset,
) -> NDArray:
    if request.param == "osrm":
        return get_enum_target_osrm_matrix(enum_df, target_df)
    if request.param == "haversine":
        return get_enum_target_haversine_matrix(enum_df, target_df)


def test_if_enum_target_cost_matrix_indices_match_enumerator_ids(
    cost_matrix: NDArray, enum_df: LocationDataset
) -> None:
    assert np.all(cost_matrix.columns.values == enum_df.get_ids())


def test_if_enum_target_cost_matrix_columns_match_target_ids(
    cost_matrix: NDArray, target_df: LocationDataset
) -> None:
    assert np.all(cost_matrix.index.values == target_df.get_ids())


def test_if_enum_target_cost_matrix_has_correct_shape(
    cost_matrix: NDArray, enum_df: LocationDataset, target_df: LocationDataset
) -> None:
    assert cost_matrix.shape == (len(target_df), len(enum_df))


def test_if_enum_target_cost_matrix_values_are_nonnegative(
    cost_matrix: NDArray,
) -> None:
    assert np.all(cost_matrix.values >= 0)
