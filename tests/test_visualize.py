from surveyscout.visualize import (
    plot_enum_targets,
    plot_assignments,
    compute_center,
)
import pytest
import pandas as pd
import numpy as np

from surveyscout.flows import basic_min_distance_flow
from surveyscout.utils import LocationDataset


@pytest.fixture(scope="module")
def assignment_df(
    enum_locs: LocationDataset, target_locs: LocationDataset
) -> pd.DataFrame:
    return basic_min_distance_flow(enum_locs, target_locs, 0, 10, 42, 500, "haversine")


def test_compute_center(
    enum_locs: LocationDataset, target_locs: LocationDataset
) -> None:
    center = compute_center(enum_locs, target_locs)
    assert np.allclose(center, (15.076699999999999, 120.71424999999999))


def test_plot_enum_targets(
    enum_locs: LocationDataset, target_locs: LocationDataset
) -> None:
    map = plot_enum_targets(enum_locs, target_locs)
    assert map is not None


def test_plot_assignments(
    enum_locs: LocationDataset,
    target_locs: LocationDataset,
    assignment_df: pd.DataFrame,
) -> None:
    map = plot_assignments(enum_locs, target_locs, assignment_df)
    assert map is not None
