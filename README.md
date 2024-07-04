# 🐦 SurveyScout 🗺️

_Assign surveyors to survey targets for minimum travel_

## What is SurveyScout?

This package takes **GPS coordinates of surveyors and survey targets** as inputs, and
outputs a table of surveyors and their assigned targets such that the **overall travel cost
(often distance) is minimized**.

## Usage

```shell
pip install git+https://github.com/IDinsight/surveyscout.git
```

We first need GPS coordinates of the surveyors and targets. Let's say we have
a pandas dataframe `enum_df`:

| enum_uid | enum_name                | gps_latitude | gps_longitude |
| -------- | ------------------------ | ------------ | ------------- |
| 0        | _African Grey Parrot_    | 82.4512      | 49.2310       |
| 1        | _Australian King Parrot_ | 78.9203      | 18.0865       |
| 2        | _Eurasian Collared Dove_ | 76.8176      | 138.6542      |
| ⋮        | ⋮                        | ⋮            | ⋮             |

and `target_df`:

| target_uid | target_name | gps_latitude | gps_longitude |
| ---------- | ----------- | ------------ | ------------- |
| 0          | _Baobab_    | 82.9514      | 138.3652      |
| 1          | _Banyan_    | 76.7823      | 160.2401      |
| 2          | _Ginkgo_    | 84.1256      | 32.7890       |
| ⋮          | ⋮           | ⋮            | ⋮             |

We create `LocationDataset` instances for surveyors and targets:

```python
from surveyscout.utils import LocationDataset

surveyors = LocationDataset(enum_df, "enum_uid", "gps_latitude","gps_longitude")
targets = LocationDataset(target_df, "target_uid", "gps_latitude", "gps_longitude")
```

You can generate assignments by running the `basic_min_distance_flow`, which will find
the assignments with minimal total distance given the constraint parameters. Here are
the constraint parameters:

- `min_target`: The minimum number of targets each surveyor is required to visit.
- `max_target`: The maximum number of targets each surveyor is allowed to visit.
- `max_cost`: The maximum cost assignable to a surveyor to visit a single target.
- `max_total_cost`: The initial maximum total cost assignable to a surveyor.

```python
from surveyscout.flows import basic_min_distance_flow

assignments = basic_min_distance_flow(
    enum_locations=surveyors,
    target_locations=targets,
    min_target=5,
    max_target=30,
    max_cost=10000,
    max_total_cost= 100000
)
```

The resulting `assignments` dataframe will look like:

| surveyor_id | target_id | cost |
| ----------- | --------- | ---- |
| 0           | 12        | 0.4  |
| 0           | 5         | 0.8  |
| 1           | 7         | 1.2  |
| 1           | 14        | 1.2  |
| 1           | 1         | 1.6  |
| ⋮           | ⋮         | ⋮    |

### More usage examples

See [example_pipeline.ipynb](https://github.com/IDinsight/surveyscout/tree/main/example_pipeline.ipynb) for more examples.

## 📏 Choosing the right travel cost function

SurveySparrow's assignment algorithms minimizes the total cost associated between the
surveyors and the assigned targets.

This cost will often be the travel distance or travel duration.

| Name        | What It Does                                                                                                                                                                                       | Caveats                                                                                                                                                                    |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `haversine` | Calculates the shortest distance between GPS coordinates.                                                                                                                                          | Quick to compute but doesn't consider road networks, terrains, or traffic.                                                                                                 |
| `osrm`      | Calculates the road distance based on OpenStreetMaps. Uses [OSRM](https://github.com/Project-OSRM/osrm-backend) at the back.                                                                       | Less expensive than `google` option but does not consider traffic or travel duration.                                                                                      |
| `google`    | Calculates travel duration considering road networks, terrain, and traffic based on [Google Maps Plaform's Distance Matrix API](https://developers.google.com/maps/documentation/distance-matrix). | Requires Google Maps Platform's API key (`GOOGLE_MAPS_PLATFORM_API_KEY`) that has access to the Distance Matrix API. Cost will be USD `n_surveyors` x `n_targets` x 0.005. |

### Using `osrm` cost function

Follow the [official
documentation](https://github.com/Project-OSRM/osrm-backend?tab=readme-ov-file#quick-start)
to run an OSRM docker container.

By default, surveyscout expects the OSRM endpoint at `http://localhost:5001` (see
`surveyscount/config.py` for default value.) If your OSRM server is at a different
endpoint, make sure to export it as environment variable:

```shell
export OSRM_URL=<your OSRM endpoint>
```

### Using `google` cost function

Use Google distance (travel duration) if you know that Google Maps works well in
the survey region.

To use `google` cost function, set your API key environment variable

```shell
export GOOGLE_MAPS_PLATFORM_API_KEY=<your Google Maps Platform API Key>
```

To save costs and simplify the calculation, SurveyScout computes only one-way
travel duration, i.e. surveyor -> target and target<sub>i</sub> ->
target<sub>j</sub> where i, j are target IDs and i < j alphabetically.
See detailed billing
[here](https://developers.google.com/maps/documentation/distance-matrix/usage-and-billing#other-usage-limits).

Since there's cost incurred for every computation of the surveyor-target travel
duration matrix, we recommend that you create a custom flow that saves and reuses the
results from `get_enum_target_google_distance_matrix` function (from
`surveyscout.tasks.compute_costs`). This is helpful when surveyors or targets get
added and you need to recompute the assignment, and when you are creating different assignments to compare them.

## Contributing

### Set up your python environment

1. Create a conda environment with python version 3.11

   ```shell
   conda create -n surveyscout python==3.11
   ```

2. Install requirements in the environment.

   ```shell
   conda activate surveyscout
   pip install -r requirements.txt
   pip install -r requirements_dev.txt
   ```

3. Install [`pre-commit`](https://pre-commit.com/).
   ```shell
   pre-commit install
   ```
