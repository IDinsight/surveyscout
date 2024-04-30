from numpy.typing import NDArray
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

    def get_ids(self) -> NDArray:
        return self.df[self.id_column].values

    def get_id_column(self) -> str:
        return self.id_column

    def get_gps_coords(self) -> NDArray:
        return self.df[[self.gps_lat_column, self.gps_lng_column]].values

    def get_gps_columns(self) -> tuple[str, str]:
        return (self.gps_lat_column, self.gps_lng_column)

    def create_subset(self, new_df: pd.DataFrame):
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


def validate_data_config(locations: LocationDataset) -> bool:
    """Checks if the enum_df is consistent with the config.

    Parameters
    ----------
    locations : class <LocationDataset>
        class containing locations informations
    config : dict
        Config

    Returns
    -------
    bool
        True if the data is consistent with the config, False otherwise.
    """
    df = locations.get_df()
    gps_columns = locations.get_gps_columns()
    assert df[locations.get_id_column()].notnull().all()
    assert df[locations.get_id_column()].unique().size == locations.get_ids().size
    assert df[gps_columns[0]].min() >= -90
    assert df[gps_columns[0]].max() <= 90
    assert df[gps_columns[1]].min() >= -180
    assert df[gps_columns[1]].max() <= 180

    return True
