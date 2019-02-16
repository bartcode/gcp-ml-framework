"""
Contains methods that transform the data in such a way that it becomes
input for a model.
"""
import tensorflow as tf
import pandas as pd

from ..utilities.config import config_key


def get_metadata():
    """
    Determines metadata of the input data.
    :return:
    """
    pass


def input_fn_csv(files_name_pattern, num_epochs, batch_size, mode, headers=None, **kwargs):
    """
    Input functions which parses CSV files.
    :param files_name_pattern: File name to TFRecords.
    :param num_epochs: Number of epochs.
    :param batch_size: Batch size.
    :param mode: Input function mode
    :param headers: Headers to use for the data.
    :return: features and label
    """
    def csv_default_value(dtype_check):
        """
        Creates a list of default values for each line of the CSV.
        :param dtype_check: dtype to convert.
        :return: List of default values (list of list)
        """
        if dtype_check.name in ['object', 'bool']:
            return ''

        return 0.  # dtype.type()

    sample_file = files_name_pattern[0] if isinstance(files_name_pattern, list) else files_name_pattern

    dtypes = kwargs.get(pd.read_csv(sample_file, nrows=1000, sep=kwargs.get('field_delim', ',')).dtypes)

    default_values = [[csv_default_value(dtype)] for dtype in dtypes]

    header_list = headers if headers else list(range(len(default_values)))

    data_set = tf.data.experimental.CsvDataset(files_name_pattern,
                                               default_values,
                                               compression_type=kwargs.get('compression_type', None),
                                               buffer_size=kwargs.get('buffer_size', None),
                                               header=kwargs.get('header', True),
                                               field_delim=kwargs.get('field_delim', ','),
                                               use_quote_delim=kwargs.get('use_quote_delim', True),
                                               na_value=kwargs.get('na_value', ''),
                                               select_cols=kwargs.get('select_cols', None))

    return data_set \
        .batch(batch_size=batch_size) \
        .repeat(count=num_epochs) \
        .shuffle(buffer_size=1 + (batch_size * 2) if mode == tf.estimator.ModeKeys.TRAIN else None) \
        .map(lambda x: dict(zip(header_list, x)))  # Map list to dictionary.


def input_fn_tfrecords(files_name_pattern, num_epochs, batch_size, mode):
    """
    Input functions which parses TFRecords.
    :param files_name_pattern: File name to TFRecords.
    :param num_epochs: Number of epochs.
    :param batch_size: Batch size.
    :param mode: Input function mode.
    :return: features and label.
    """
    return tf.contrib.data.make_batched_features_dataset(
        file_pattern=files_name_pattern,
        batch_size=batch_size,
        features=get_metadata().schema.as_feature_spec(),  # This doesn't work yet.
        reader=tf.data.TFRecordDataset,
        num_epochs=num_epochs,
        shuffle=True if mode == tf.estimator.ModeKeys.TRAIN else False,
        shuffle_buffer_size=1 + (batch_size * 2),
        prefetch_buffer_size=1,
    )


def input_fn(files_name_pattern,
             num_epochs=None,
             batch_size=200,
             mode=tf.estimator.ModeKeys.TRAIN,
             input_format='tfrecords',
             **kwargs):
    """
    General input function which parses csv, json, or tfrecords.
    :param files_name_pattern: File name pattern.
    :param num_epochs: Number of epochs.
    :param batch_size: Batch size.
    :param mode: Input function mode.
    :param input_type: Input type: csv or tfrecords
    :param kwargs: Other arguments to the input functions.
    :return: features and label
    """
    input_fn_reference = input_fn_csv if input_format == 'csv' else input_fn_tfrecords

    data_set = input_fn_reference(files_name_pattern, num_epochs, batch_size, mode, **kwargs)

    iterator = data_set.make_one_shot_iterator()

    features = iterator.get_next()

    label = features.pop(config_key('model.label'))

    if mode == tf.estimator.ModeKeys.PREDICT:
        return features

    return features, label


def json_serving_input_fn():
    """
    Build the serving inputs.
    """
    inputs = {}

    # TODO: Add metadata.
    # for feature, value in METADATA.schema.column_schemas.items():
    #     inputs[feature] = tf.placeholder(
    #         shape=[None],
    #         dtype=value.domain.dtype
    #     )

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
