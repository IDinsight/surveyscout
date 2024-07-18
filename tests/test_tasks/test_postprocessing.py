import pandas as pd
import pytest
from numpy.typing import NDArray

from surveyscout.utils import LocationDataset
from surveyscout.tasks.compute_cost import (
    get_enum_target_haversine_matrix,
)
from surveyscout.tasks.models import min_target_optimization_model
from surveyscout.tasks.postprocessing import convert_assignment_matrix_to_table

params = [
    [0, 10, 42, 500],
    [2, 4, 35, 300],
]


@pytest.fixture(scope="module")
def enum_target_cost_matrix(
    enum_locs: LocationDataset, target_locs: LocationDataset
) -> NDArray:
    return get_enum_target_haversine_matrix(
        enum_locations=enum_locs, target_locations=target_locs
    )


@pytest.fixture(scope="module")
def assignment_matrix(enum_target_cost_matrix):
    return min_target_optimization_model(enum_target_cost_matrix.values, 0, 10, 42, 500)


def test_postprocess_results(
    assignment_matrix, enum_locs, target_locs, enum_target_cost_matrix
):
    df = convert_assignment_matrix_to_table(
        assignment_matrix, enum_locs, target_locs, enum_target_cost_matrix
    )

    assert "target_id" in df.columns, "The column target_id is missing"
    assert "enum_id" in df.columns, "The column enum_id is missing"
    assert "cost" in df.columns, "The column cost is missing"

    assert df.shape[0] == len(target_locs)
    assert df.shape[1] == len(enum_locs)
    assert (
        not df[["target_id", "enum_id"]].duplicated().any()
    ), "The combination of col1 and col2 is not unique"

    assignment_df = pd.DataFrame(
        assignment_matrix,
        index=target_locs.get_ids(),
        columns=enum_locs.get_ids(),
    )
    for _, row in df.iterrows():
        assert (
            assignment_df.loc[row["target_id"], row["enum_id"]] == 1
        ), "The assignment matrix is not consistent with the postprocessed results"
        assert (
            row["cost"] == enum_target_cost_matrix.loc[row["target_id"], row["enum_id"]]
        ), "The cost is not consistent with the cost matrix"
