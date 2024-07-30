"""Test configuration."""

from pathlib import Path
from typing import Generator
import numpy as np
from numpy.typing import NDArray
import aiohttp

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
def patch_osrm_table_call(monkeysession: pytest.MonkeyPatch) -> None:
    """
    Monkeypatch calls to OSRM /table API
    """

    monkeysession.setattr(
        "surveyscout.tasks.compute_cost.osrm._get_enum_target_matrix_osrm",
        mock_return_matrix,
    )


async def mock_fetch_osrm_trip_data(
    url: str, coordinates_string: str, session: aiohttp.ClientSession
) -> dict:
    """Mock OSRM trip data"""
    lng_lat_coords = [
        list(map(float, x.split(","))) for x in coordinates_string.split(";")
    ]

    order = np.random.permutation(len(lng_lat_coords))
    waypoints = [
        {
            "waypoint_index": i,
            "trips_index": 0,
            "distance": 2.652005364,
            "name": "",
            "location": coords,
        }
        for i, coords in zip(order, lng_lat_coords)
    ]
    legs = [
        {
            "steps": [],
            "summary": "",
            "weight": 1178.4,
            "duration": 1178.4,
            "distance": 12450.5,
        }
        for i in range(len(lng_lat_coords))
    ]

    return {
        "code": "Ok",
        "trips": [
            {
                "legs": legs,
                "weight_name": "routability",
                "weight": 8910.4,
                "duration": 8818.3,
                "distance": 22183.3,
            }
        ],
        "waypoints": waypoints,
    }


@pytest.fixture(scope="session", autouse=True)
def patch_osrm_trip_call(monkeysession: pytest.MonkeyPatch) -> None:
    """
    Monkeypatch calls to OSRM /trip API
    """

    monkeysession.setattr(
        "surveyscout.tasks.rank._fetch_osrm_trip_data",
        mock_fetch_osrm_trip_data,
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
