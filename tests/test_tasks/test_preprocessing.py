"""
Test cost computation functions.
"""

import numpy as np
import pytest

from surveyscout.tasks.preprocessing import get_enum_target_haversine_matrix


@pytest.fixture(scope="module")
def enum_target_haversine_matrix(enum_df, target_df):
    return get_enum_target_haversine_matrix(
        enum_locations=enum_df, target_locations=target_df
    )


def test_if_enum_target_cost_matrix_indices_match_enumerator_ids(
    enum_target_haversine_matrix, enum_df
):
    assert np.all(enum_target_haversine_matrix.columns.values == enum_df.get_ids())


def test_if_enum_target_cost_matrix_columns_match_target_ids(
    enum_target_haversine_matrix, target_df
):
    assert np.all(enum_target_haversine_matrix.index.values == target_df.get_ids())


def test_if_enum_target_cost_matrix_has_correct_shape(
    enum_target_haversine_matrix, enum_df, target_df
):
    assert enum_target_haversine_matrix.shape == (len(target_df), len(enum_df))


def test_if_enum_target_cost_matrix_values_are_nonnegative(
    enum_target_haversine_matrix,
):
    assert np.all(enum_target_haversine_matrix.values >= 0)
