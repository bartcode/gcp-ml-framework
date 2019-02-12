"""
Methods to read files.
"""
import json
import os
import yaml


def load_config(file_name):
    """
    Load configuration file, either YAML or JSON.
    :param file_name: Path to file
    :return: Dict
    """
    file_base = os.path.basename(file_name)

    # Detect file extension
    extension = file_base[file_base.rindex('.') + 1:].lower() \
        if '.' in file_base \
        else None

    with open(file_name, 'r') as file_stream:
        if extension == 'json':
            return json.loads(file_stream.read())
        elif extension in ['yml', 'yaml']:
            return yaml.load(file_stream)
