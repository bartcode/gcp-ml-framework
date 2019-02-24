# GCP ML Framework
Framework for large scale data processing, (DNN) model training, and prediction delivery
on the Google Cloud Platform.

## Installation
```bash
$ virtualenv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```

## Usage
```bash
$ ./runner.sh (preprocess|tune|train|predict) (local|cloud)
```

## Notes
- This framework uses Python 2.7 for the _sole_ reason that Beam
    is not available in Python 3.x.
