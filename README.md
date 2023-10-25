# optiver2023

## Download data and compeition environment files

To use this repository,

Download the folder 'optiver-trading-at-the-close' from https://www.kaggle.com/competitions/optiver-trading-at-the-close/data, and add it to the root of the repository.

Its structure look like
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