from math import radians
from pathlib import Path

import pandas as pd
import yamlx
from sklearn.metrics.pairwise import haversine_distances


project_root = Path(__file__).parents[1]


class LocationDataset(object):
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


def get_intergroup_haversine_matrix(row_locations, col_locations):
    """
    Computes the haversine distance between each pair of enumerator and target.

    Parameters
    ----------
    row_locations : class <LocationDataset>
        class containing locations informations
    col_locations : class <LocationDataset>
        class containing locations informations
    Returns
    -------
    pandas.DataFrame
        Haversine distance matrix
    """
    row_ids = row_locations.get_ids()
    row_gps_coords = row_locations.get_gps_coords()
    col_ids = col_locations.get_ids()
    col_gps_coords = col_locations.get_gps_coords()
    row_coords_rads = pd.DataFrame(row_gps_coords).applymap(radians).values
    col_coords_rads = pd.DataFrame(col_gps_coords).applymap(radians).values

    distance_array = EARTH_RADIUS * haversine_distances(
        row_coords_rads, col_coords_rads
    )

    distance_df = pd.DataFrame(distance_array, columns=col_ids, index=row_ids)

    return distance_df
