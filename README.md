# GCP ML Framework
Framework for large scale data processing, (DNN) model training, and prediction delivery
on the Google Cloud Platform.

## Installation
```bash
$ python3 -m venv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```

## Usage
```bash
$ ./runner.sh (preprocess|tune|train|predict) (local|cloud)
```

## Notes
- Support for Apache Beam in Python 3 is still experimental.
