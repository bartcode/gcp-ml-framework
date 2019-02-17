"""
Methods to read files.
"""
import itertools
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

    return {}


def config_key(path_key, config=load_config('./config.yml')):
    """
    Load key from config file.
    :param path_key: Dot-separated path to value in dictionary, such as: model.train
    :param config: Configuration dictionary.
    :return: Config value
    """
    if '.' in path_key:
        base, sub_path = tuple(path_key.split('.', 1))

        return config_key(sub_path, config[base])

    return config.get(path_key, None)


def config_path(*args):
    """
    Join multiple paths from configuration and always use config['path']['base'] as base path.
    Use as follows:
    ```python
    config_path('model.train')
    ```
    :param args: Lists of keys to load.
    :return: Path to file
    """

    def add_base_path(items, base_path=config_key('path.base')):
        """
        If the key in the configuration file leads to a list, the base path
        will be added to all of them.
        :param items: List of paths.
        :param base_path: Base path to prepend.
        :return: List of paths prepended with the base path.
        """
        if isinstance(items, list):
            return [os.path.join(base_path, item)
                    for item in items]

        # If the input wasn't a list, return a list with that single item.
        return [items]

    paths = [config_key(path) for path in args]

    path_bases = list(itertools.chain(*[add_base_path(path) for path in paths]))

    # Return a single value if there's only one value to return.
    if len(path_bases) == 1:
        return path_bases[0]

    return path_bases
