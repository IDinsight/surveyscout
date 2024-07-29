from typing import List
import aiohttp
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import requests
import asyncio

from surveyscout.config import OSRM_URL
from surveyscout.utils import LocationDataset


def get_enum_target_osrm_matrix_async(
    enum_locations: LocationDataset,
    target_locations: LocationDataset,
) -> pd.DataFrame:
    """Get the matrix of distances between enumerators and targets using OSRM api.
    Make asynchronous requests to the OSRM API.

    This function calls the OSRM /table/v1/driving/ api endpoint to get the matrix
    of distances between enumerators and targets. The distance represents the
    distance of the fastest route between the two points.

    Parameters
    ----------
    enum_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of enumerators.

    target_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of targets,
         with a similar structure to `enum_locations`.

    Returns
    -------
    pd.DataFrame
        distance matrix between enumerators and targets.
        Columns are enumerator IDs, rows are target IDs.
    """
    enums_coords = enum_locations.get_gps_coords()
    targets_coords = target_locations.get_gps_coords()

    url = OSRM_URL + "/table/v1/driving/"
    matrix = _get_enum_target_matrix_osrm_async(url, targets_coords, enums_coords)
    matrix_df = pd.DataFrame(
        matrix, index=target_locations.get_ids(), columns=enum_locations.get_ids()
    )
    return matrix_df


def _get_enum_target_matrix_osrm_async(
    url: str, target_coords: NDArray, enum_coords: NDArray
) -> List:
    """Get the matrix of distances between enumerators and targets using OSRM api."""

    matrix = asyncio.run(_gather_matrix(url, target_coords, enum_coords))
    return matrix


async def _gather_matrix(url: str, target_coords: NDArray, enum_coords: NDArray):
    matrix_computation_tasks = [
        _get_unique_target_enum_row_osrm_async(url, target_coord, enum_coords)
        for target_coord in target_coords
    ]
    return await asyncio.gather(*matrix_computation_tasks)


async def _get_unique_target_enum_row_osrm_async(
    url: str, target_coord: NDArray, enum_coords: NDArray
) -> NDArray | List:
    """Get the row of the matrix for a target coordinate."""
    coords = np.insert(enum_coords, 0, target_coord, axis=0)
    url = _format_url_with_coords_osrm(url, coords)

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response_json = await response.json()

    if "distances" in response_json:
        data = response_json["distances"][0][1:]
        return np.array(data) / 1000
    else:
        return []


def _format_url_with_coords_osrm(url: str, coords: NDArray) -> str:
    """Formats URL with GPS coordinates for OSRM API"""
    coord_str = ";".join([f"{x[1]},{x[0]}" for x in coords])
    url = url + coord_str + "?sources=0&annotations=distance,duration"
    return url


def get_enum_target_osrm_matrix(
    enum_locations: LocationDataset,
    target_locations: LocationDataset,
) -> pd.DataFrame:
    """Get the matrix of distances between enumerators and targets using OSRM api.

    This function calls the OSRM /table/v1/driving/ api endpoint to get the matrix
    of distances between enumerators and targets. The distance represents the
    distance of the fastest route between the two points.

    Parameters
    ----------
    enum_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of enumerators.

    target_locations : class <LocationDataset>
        A <LocationDataset> object containing the id and locations of targets,
         with a similar structure to `enum_locations`.

    Returns
    -------
    pd.DataFrame
        distance matrix between enumerators and targets.
        Columns are enumerator IDs, rows are target IDs.
    """
    enums_coords = enum_locations.get_gps_coords()
    targets_coords = target_locations.get_gps_coords()

    url = OSRM_URL + "/table/v1/driving/"
    matrix = _get_enum_target_matrix_osrm(url, targets_coords, enums_coords)
    matrix_df = pd.DataFrame(
        matrix, index=target_locations.get_ids(), columns=enum_locations.get_ids()
    )
    return matrix_df


def _get_enum_target_matrix_osrm(
    url: str, target_coords: NDArray, enum_coords: NDArray
) -> NDArray | List:
    """Get the matrix of distances between enumerators and targets
    using OSRM."""
    matrix = [
        _get_unique_target_enum_row_osrm(url, target_coord, enum_coords)
        for target_coord in target_coords
    ]

    return matrix


def _get_unique_target_enum_row_osrm(
    url: str, target_coord: NDArray, enum_coords: NDArray
) -> NDArray | List:
    """Get the row of the matrix for a target coordinate."""
    coords = np.insert(enum_coords, 0, target_coord, axis=0)
    url = _format_url_with_coords_osrm(url, coords)
    response = requests.get(url)

    if "distances" in response.json():
        data = response.json()["distances"][0][1:]
        return np.array(data) / 1000
    else:
        return []
