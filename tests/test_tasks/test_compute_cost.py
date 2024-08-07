import numpy as np
import pandas as pd
import pytest
from surveyscout.tasks.compute_cost import (
    get_enum_target_haversine_matrix,
    get_enum_target_osrm_matrix,
    get_enum_target_google_distance_matrix,
)
from surveyscout.utils import LocationDataset

"""
Test cost computation functions.
"""


@pytest.fixture(scope="module")
def enum_target_haversine_matrix(
    enum_locs: LocationDataset, target_locs: LocationDataset
) -> pd.DataFrame:
    return get_enum_target_haversine_matrix(
        enum_locations=enum_locs, target_locations=target_locs
    )


@pytest.fixture(scope="module")
def enum_target_osrm_matrix(
    enum_locs: LocationDataset, target_locs: LocationDataset
) -> pd.DataFrame:
    return get_enum_target_osrm_matrix(
        enum_locations=enum_locs, target_locations=target_locs
    )


@pytest.fixture(scope="module")
def enum_target_google_distance_matrix(
    enum_locs: LocationDataset, target_locs: LocationDataset
) -> pd.DataFrame:
    return get_enum_target_google_distance_matrix(
        enum_locations=enum_locs, target_locations=target_locs
    )


@pytest.fixture(params=["osrm", "haversine", "google"])
def cost_matrix(
    request: pytest.FixtureRequest,
    enum_locs: LocationDataset,
    target_locs: LocationDataset,
) -> pd.DataFrame:
    if request.param == "osrm":
        return get_enum_target_osrm_matrix(enum_locs, target_locs)
    if request.param == "haversine":
        return get_enum_target_haversine_matrix(enum_locs, target_locs)
    if request.param == "google":
        return get_enum_target_google_distance_matrix(enum_locs, target_locs)


def test_if_enum_target_cost_matrix_indices_match_enumerator_ids(
    cost_matrix: pd.DataFrame, enum_locs: LocationDataset
) -> None:
    assert np.all(cost_matrix.columns.values == enum_locs.get_ids())


def test_if_enum_target_cost_matrix_columns_match_target_ids(
    cost_matrix: pd.DataFrame, target_locs: LocationDataset
) -> None:
    assert np.all(cost_matrix.index.values == target_locs.get_ids())


def test_if_enum_target_cost_matrix_has_correct_shape(
    cost_matrix: pd.DataFrame, enum_locs: LocationDataset, target_locs: LocationDataset
) -> None:
    assert cost_matrix.shape == (len(target_locs), len(enum_locs))


def test_if_enum_target_cost_matrix_values_are_nonnegative(
    cost_matrix: pd.DataFrame,
) -> None:
    assert np.all(cost_matrix.values >= 0)
