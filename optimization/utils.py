import numpy as np
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parents[1]


class LocationDataset(object):
    """A container for managing and accessing location-related data within a pandas DataFrame."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        id_column: str,
        gps_lat_column: str,
        gps_lng_column: str,
    ):
        self.df = dataframe
        self.id_column = id_column
        self.gps_lat_column = gps_lat_column
        self.gps_lng_column = gps_lng_column
        self._data_len = len(dataframe)

    def get_ids(self):
        return self.df[self.id_column].values

    def get_id_column(self):
        return self.id_column

    def get_gps_coords(self):
        return self.df[[self.gps_lat_column, self.gps_lng_column]].values

    def get_gps_columns(self):
        return (self.gps_lat_column, self.gps_lng_column)

    def create_subset(self, new_df):
        return LocationDataset(
            dataframe=new_df,
            id_column=self.id_column,
            gps_lat_column=self.gps_lat_column,
            gps_lng_column=self.gps_lng_column,
        )

    def get_df(self):
        return self.df

    def __len__(self):
        return len(self.df)


def validate_data_config(data):
    """Checks if the enum_df is consistent with the config.

    Parameters
    ----------
    data : class <LocationDataset>
        class containing locations informations
    config : dict
        Config

    Returns
    -------
    bool
        True if the data is consistent with the config, False otherwise.
    """
    df = data.get_df()
    gps_columns = data.get_gps_columns()
    assert df[data.get_id_column()].notnull().all()
    assert df[data.get_id_column()].unique().size == data.get_ids().size
    assert df[gps_columns[0]].min() >= -90
    assert df[gps_columns[0]].max() <= 90
    assert df[gps_columns[1]].min() >= -180
    assert df[gps_columns[1]].max() <= 180

    return True


def haversine(lat1, lon1, lat2, lon2):
    """Compute the haversine distance between two GPS coordinates."""
    R = 6371.0

    # convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    return distance


def get_percentile_distance(cost_matrix, max_perc):
    """Get the maximum distance from the cost matrix by computing percentile ."""
    max_distance = np.percentile(cost_matrix, max_perc)
    return max_distance


def format_coords_into_string_google(coords):
    """Formats GPS coordinates into string as required by Google Distance Matrix API

    Parameters
    ----------
    coords : np.array
        numpy array of shape (2, n) representing latitude longitude pairs

    Returns
    -------
    str
        formatted string of GPS coordinates
    """
    return "|".join(map(lambda x: str(x[0]) + "," + str(x[1]), coords))


def format_url_with_coords_osrm(url, coords):
    """Formats URL with GPS coordinates for OSRM API"""
    coord_str = ";".join([f"{x[0]},{x[1]}" for x in coords])
    url = url + coord_str + "?sources=0&annotations=distance,duration"
    return url
