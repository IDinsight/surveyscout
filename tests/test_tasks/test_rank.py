import pandas as pd
import pytest
from pathlib import Path

from surveyscout.utils import LocationDataset
from surveyscout.tasks.compute_cost import (
    get_enum_target_haversine_matrix,
)
from surveyscout.tasks.models import min_target_optimization_model
from surveyscout.tasks.postprocessing import convert_assignment_matrix_to_table
from surveyscout.tasks.rank import add_visit_order


def enum_ids():
    """Load test enumerator file."""
    df = pd.read_csv(Path(__file__).parents[1] / "data/test_enum_data.csv")
    return df["id"].tolist()


@pytest.fixture(scope="module")
def assignment_df(enum_locs: LocationDataset, target_locs: LocationDataset):
    enum_target_cost_matrix = get_enum_target_haversine_matrix(
        enum_locations=enum_locs, target_locations=target_locs
    )
    assignment_matrix = min_target_optimization_model(
        enum_target_cost_matrix.values, 0, 10, 42, 500
    )
    return convert_assignment_matrix_to_table(
        assignment_matrix, enum_locs, target_locs, enum_target_cost_matrix
    )


def test_add_visit_order(assignment_df, target_locs):
    df = add_visit_order(assignment_df, target_locs)
    assert "target_visit_order" in df.columns, "The column visit_order is missing"
    assert (
        "distance_to_next_in_km" in df.columns
    ), "The column distance_to_next_in_km is missing"
    assert df.shape[0] == len(target_locs)


@pytest.mark.parametrize("enum_id", enum_ids())
def test_target_visit_order_is_valid_for_each_enum(enum_id, assignment_df, target_locs):
    df = add_visit_order(assignment_df, target_locs)

    subdf = df[df["enum_id"] == enum_id]

    assert subdf["target_visit_order"].min() == 0
    assert subdf["target_visit_order"].max() == len(subdf) - 1
    assert (
        not subdf["target_visit_order"].duplicated().any()
    ), f"The visit order is not unique for enumerator {enum_id}"
