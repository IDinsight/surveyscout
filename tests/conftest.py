"""Test configuration."""

from pathlib import Path
from typing import Generator
import numpy as np
from numpy.typing import NDArray

import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch

from surveyscout.utils import LocationDataset


@pytest.fixture(scope="session")
def monkeypatch():
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="session")
def enum_df():
    """Load test enumerator file."""
    df = pd.read_csv(Path(__file__).parent / "data/test_enum_data.csv")
    data = LocationDataset(
        dataframe=df,
        id_column="id",
        gps_lat_column="gps_lat",
        gps_lng_column="gps_lon",
    )
    return data


@pytest.fixture(scope="session")
def target_df():
    """Load test target file."""
    df = pd.read_csv(Path(__file__).parent / "data/test_target_data.csv")
    data = LocationDataset(
        dataframe=df,
        id_column="id",
        gps_lat_column="gps_lat",
        gps_lng_column="gps_lon",
    )
    return data


@pytest.fixture(scope="session")
def monkeysession(
    request: pytest.FixtureRequest,
) -> Generator[pytest.MonkeyPatch, None, None]:
    from _pytest.monkeypatch import MonkeyPatch

    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


def mock_return_matrix(
    url: str,
    target_coord: NDArray,
    enum_coord: NDArray,
) -> NDArray:
    """Mock return matrix."""
    n, m = len(target_coord), len(enum_coord)
    return np.linspace(0, 20, num=n * m).reshape(n, m)


@pytest.fixture(scope="session", autouse=True)
def patch_osrm_call(monkeysession: pytest.MonkeyPatch) -> None:
    """
    Monkeypatch call OSRM API.
    """

    monkeysession.setattr(
        "surveyscout.tasks.compute_cost.osrm._get_enum_target_matrix_osrm",
        mock_return_matrix,
    )
