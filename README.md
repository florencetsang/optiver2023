# optiver2023

## Download data and competition environment files

To use this repository,

Download the folder 'optiver-trading-at-the-close' from https://www.kaggle.com/competitions/optiver-trading-at-the-close/data, and add it to the root of the repository.

Its structure looks like
```
.
├── optiver-trading-at-the-close/
│   ├── example_test_files/
│   │   ├── revealed_targets.csv
│   │   ├── sample_submission.csv
│   │   └── test.csv
│   ├── optiver2023/
│   │   ├── __init__.py
│   │   └── competition.cpython-310-x86_64-linux-gnu.so
│   ├── public_timeseries_testing_util.py
│   └── train.csv
├── README.md
└── (other files)
```

The reason it was not added to this repository was because "optiver-trading-at-the-close/train.csv is 611.21 MB; this exceeds GitHub's file size limit of 100.00 MB".

It contains competition environment files and data files.

## Development environment setup guide

Environment setup guide: 
[setup.md](setup.md)

## Experiment running 

To run experiments of different models and different feature sets:

First, cd to optiver2023/florence/
Then, running the following:

`python optuna_main.py <model_name> <model_type>`

eg: `python optuna_main.py 20240422_lgb_moving_avg lgb`




