"""
Methods to read files.
"""
import itertools
import json
import os

import tensorflow as tf
import yaml


def load_config(file_name):
    """
    Load configuration file, either YAML or JSON.
    :param file_name: Path to file
    :return: Dict
    """
    file_base = os.path.basename(file_name)

    # Detect file extension
    extension = file_base[file_base.rindex('.') + 1:].lower() if '.' in file_base else None

    with open(file_name, 'r') as file_stream:
        if extension == 'json':
            return json.loads(file_stream.read())
        elif extension in ['yml', 'yaml']:
            return yaml.load(file_stream.read(), Loader=yaml.SafeLoader)
        return {}


def config_key(path_key, config=None):
    """
    Load key from config file.
    :param path_key: Dot-separated path to value in dictionary, such as: model.train
    :param config: Configuration dictionary.
    :return: Config value
    """
    config = config \
        if config \
        else load_config('./config.yml')

    if '.' in path_key:
        base, sub_path = tuple(path_key.split('.', 1))

        return config_key(sub_path, config=config[base])

    return config.get(path_key, None)


def config_path(*args, **kwargs):
    """
    Join multiple paths from configuration and always use config['path']['base'] as base path.
    Use as follows:
    ```python
    config_path('model.train')
    ```
    :param args: Lists of keys to load.
    :param kwargs: Keyword arguments. Currently available key: `base_path`. Which path to use as a base path.
    :return: Path to file
    """
    config_key_path = config_key('path.base', config=kwargs.get('config'))

    def add_base_path(items, base_path=config_key_path):
        """
        If the key in the configuration file leads to a list, the base path
        will be added to all of them.
        :param items: List of paths.
        :param base_path: Base path to prepend.
        :return: List of paths prepended with the base path.
        """
        cloud_path = config_key('cloud.bucket', config=kwargs.get('config')) \
            if kwargs.get('config') and kwargs.get('config').get('execution') == 'cloud' \
            else './'

        if isinstance(items, list):
            return [os.path.join(cloud_path, base_path, item)
                    for item in items]

        # If the input wasn't a list, return a list with that single item.
        return [os.path.join(cloud_path, base_path, items)]

    paths = [config_key(path, config=kwargs.get('config')) for path in args]

    path_bases = list(itertools.chain(
        *[
            add_base_path(path,
                          base_path=kwargs.get('base_path', config_key('path.base',
                                                                       config=kwargs.get('config'))))
            for path in paths
        ]
    ))

    # Return a single value if there's only one value to return.
    if len(path_bases) == 1 or kwargs.get('max_len') == 1:
        return path_bases[0]

    if kwargs.get('max_len'):
        return path_bases[:int(kwargs.get('max_len'))]

    return path_bases


def get_tensorflow_session_config():
    """
    Determines tf.ConfigProto from environment variables.
    :return: TensorFlow configuration (tf.ConfigProto).
    """
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    try:
        if tf_config['task']['type'] == 'master':
            # The master communicates with itself
            # and the ps (parameter server).
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:%d' % tf_config['task']['index']])
    except KeyError:
        pass

    return tf.ConfigProto()
