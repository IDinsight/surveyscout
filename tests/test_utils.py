import numpy as np
import pandas as pd
import pytest

from surveyscout.utils import validate_data_config, LocationDataset


@pytest.fixture
def data():
    df = pd.DataFrame(
        {
            "id": np.arange(10),
            "gps_lat": np.random.uniform(-90, 90, 10),
            "gps_lng": np.random.uniform(-180, 180, 10),
        }
    )
    data = LocationDataset(
        dataframe=df, id_column="id", gps_lat_column="gps_lat", gps_lng_column="gps_lng"
    )
    return data


@pytest.mark.parametrize(
    ("lat", "lng"),
    [
        (np.inf, 0),
        (0, np.inf),
        (-np.inf, 0),
        (0, -np.inf),
        (90.1, -180),
        (-90.1, 180),
        (90, -180.1),
        (90, 180.1),
    ],
)
def test_data_config_validation_fails_with_out_of_range_gps_coords(lat, lng, data):
    gps_cols = data.get_gps_columns()
    data.get_df().loc[0, gps_cols] = (lat, lng)

    with pytest.raises(AssertionError):
        validate_data_config(data)


def test_data_config_validation_fails_with_empty_id(data):
    data.get_df().loc[0, data.get_id_column()] = None

    with pytest.raises(AssertionError):
        validate_data_config(data)


def test_data_config_validation_fails_with_duplicate_ids(data):
    df = data.get_df()
    df.loc[0, data.get_id_column()] = df.loc[1, data.get_id_column()]

    with pytest.raises(AssertionError):
        validate_data_config(data)


def test_data_config_validation_succeeds(data):
    assert validate_data_config(data)
