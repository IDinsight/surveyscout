# 🐦 SurveyScout 🗺️

_Assign surveyors to survey targets for minimum travel_

## What is SurveyScout?

This package takes **GPS coordinates of surveyors and survey targets** as inputs, and
outputs a table of surveyors and their assigned targets such that the **overall travel cost
(often distance) is minimized**.

## Usage

```shell
pip install https://github.com/IDinsight/surveyscout.git`
```

We first need GPS coordinates of the surveyors and targets. Let's say we have
a pandas dataframe `enum_df`:

|enum_uid|enum_name|gps_latitude|gps_longitude|
|--|--|--|--|
|0|_African Grey Parrot_| 82.4512  | 49.2310   |
|1|_Australian King Parrot_| 78.9203  | 18.0865   |
|2|_Eurasian Collared Dove_| 76.8176  | 138.6542  |
|⋮|⋮|⋮|⋮|

and `target_df`:
|target_uid|target_name|gps_latitude|gps_longitude|
|--|--|--|--|
|0|_Baobab_| 82.9514  | 138.3652  |
|1|_Banyan_| 76.7823  | 160.2401  |
|2|_Ginkgo_| 84.1256  | 32.7890   |
|⋮|⋮|⋮|⋮|

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
- `max_distance`: The maximum distance assignable to a surveyor to visit a single target.
- `max_total_distance`: The initial maximum total distance assignable to a surveyor.

```python
from surveyscout.flows import basic_min_distance_flow

assignments = basic_min_distance_flow(
    enum_locations=surveyors, 
    target_locations=targets,
    min_target= 5,
    max_target= 30, 
    max_distance=10000, 
    max_total_distance= 100000
)
```

The resulting `assignments` dataframe will look like:

| surveyor_id | target_id | cost |
|---------------|-----------|------|
| 0             | 12        | 0.4  |
| 0             | 5         | 0.8  |
| 1             | 7         | 1.2  |
| 1             | 14        | 1.2  |
| 1             | 1         | 1.6  |
| ⋮             | ⋮          | ⋮    |

### More usage examples

See [example_pipeline.ipynb](https://github.com/IDinsight/surveyscout/tree/main/example_pipeline.ipynb) for more examples.

## Contributing

### Set up your python environment

1. Create a conda environment with python version 3.11

    ```shell
    conda create -n surveyscout python==3.11
    ```

2. Install requirements in the environment. You must install [`pre-commit`](https://pre-commit.com/) as well.

    ```shell
    conda activate surveyscout
    pip install -r requirements.txt
    pip install -r requirements_dev.txt
    ```
