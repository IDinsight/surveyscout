"""Test configuration."""

from pathlib import Path

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
