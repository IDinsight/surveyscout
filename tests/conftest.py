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
def enum_locs():
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
def target_locs():
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


success_element = {
    "distance": {"text": "23.8 km", "value": 23829},
    "duration": {"text": "44 mins", "value": 2629},
    "status": "OK",
}


def mock_get_google_distance_matrix(origins: NDArray, destinations: NDArray) -> dict:
    """Mock Google Distance Matrix API response."""
    n_orig = len(origins)
    n_dest = len(destinations)
    elements = [success_element] * n_dest
    rows = [{"elements": elements}] * n_orig
    return {"rows": rows, "status": "OK"}


@pytest.fixture(scope="session", autouse=True)
def patch_google_distance_matrix(monkeysession: pytest.MonkeyPatch) -> None:
    """
    Monkeypatch call Google Distance Matrix API.
    """
    monkeysession.setattr(
        "surveyscout.tasks.compute_cost.google_distance_matrix._get_google_distance_matrix",
        mock_get_google_distance_matrix,
    )
