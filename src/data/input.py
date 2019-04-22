"""
Contains methods that transform the data in such a way that it becomes
input for a model.
"""
from glob import glob

import pandas as pd
import tensorflow as tf
from pandas.api.types import is_string_dtype  # , is_int64_dtype
from tensorflow_transform.tf_metadata import metadata_io, dataset_metadata, dataset_schema

from ..utilities.config import config_key, config_path


def _get_single_train_file(**kwargs):
    """
    Obtain a single file path used for training.
    :param kwargs: Other keyword arguments.
    :return: File path
    """
    train_file_config = config_path('path.train-files', config=kwargs.get('config'))

    if kwargs.get('config', {}).get('execution') == 'cloud':
        return train_file_config[0] \
            if isinstance(train_file_config, list) \
            else train_file_config

    return glob(train_file_config[0])[0] \
        if isinstance(train_file_config, list) \
        else glob(train_file_config)[0]


def get_headers(file_name, **kwargs):
    """
    Retrieves header columns of a file.
    :param file_name: Path to file
    :param kwargs: Other keyword arguments.
    :return: List of header names
    """
    dtypes = pd.read_csv(file_name, nrows=5, sep=config_key('path.field-delim', config=kwargs.get('config'))).dtypes

    return dtypes.index.tolist() \
        if dtypes.index.tolist() \
        else list(range(len(dtypes)))


def get_metadata(file_name, **kwargs):
    """
    Determines metadata of the input data.
    :param file_name: Name of CSV file to parse.
    :param kwargs: Other keyword arguments.
    :return: DatasetMetadata
    """

    def dtype_to_metadata(dtype):
        """
        Converts Pandas Dataframe dtype to TF dataset metadata.
        :param dtype: Pandas dtype.
        :return: tf.FixedLenFeature with corresponding data type.
        """
        if is_string_dtype(dtype):
            return tf.FixedLenFeature([], tf.string)

        # # By commenting the two lines below, every column
        # # is forced to be float32 instead of int64.
        # if is_int64_dtype(dtype):
        #     return tf.FixedLenFeature([], tf.int64)

        return tf.FixedLenFeature([], tf.float32)

    if config_key('model.input-format', kwargs.get('config')) == 'tfrecords':
        return metadata_io.read_metadata(config_key('path.metadata', config=kwargs.get('config')))

    try:
        # It's actually best to define the metadata schema manually,
        # but this is a good start.
        return dataset_metadata.DatasetMetadata(
            dataset_schema.from_feature_spec({
                name: dtype_to_metadata(dtype)
                for name, dtype in pd.read_csv(file_name,
                                               nrows=1000,
                                               sep=config_key('path.field-delim',
                                                              config=kwargs.get('config'))).dtypes.to_dict().items()
            }))
    except KeyError:
        return dataset_metadata.DatasetMetadata(
            dataset_metadata.DatasetMetadata(dataset_schema.from_feature_spec({}))
        )


def input_fn_csv(files_name_pattern, num_epochs, batch_size, mode, **kwargs):
    """
    Input functions which parses CSV files.
    :param files_name_pattern: File name to TFRecords.
    :param num_epochs: Number of epochs.
    :param batch_size: Batch size.
    :param mode: Input function mode
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

    dtypes = pd.read_csv(_get_single_train_file(), nrows=1000,
                         sep=config_key('path.field-delim', config=kwargs.get('config'))).dtypes

    default_values = [[csv_default_value(dtype)] for dtype in dtypes]

    header_list = dtypes.index.tolist() \
        if dtypes.index.tolist() \
        else list(range(len(default_values)))

    data_set = tf.data.experimental.CsvDataset(
        filenames=files_name_pattern,
        record_defaults=default_values,
        compression_type=kwargs.get('compression_type', None),
        buffer_size=kwargs.get('buffer_size', None),
        header=kwargs.get('header', True),
        field_delim=config_key('path.field-delim', config=kwargs.get('config')),
        use_quote_delim=kwargs.get('use_quote_delim', True),
        na_value=kwargs.get('na_value', ''),
        select_cols=kwargs.get('select_cols', None)
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        return data_set \
            .batch(batch_size=batch_size) \
            .repeat(count=num_epochs) \
            .shuffle(buffer_size=1 + (batch_size * 2)) \
            .map(lambda *x: dict(zip(header_list, x)))  # Map list to dictionary.

    # Eval mode
    return data_set \
        .batch(batch_size=batch_size) \
        .repeat(count=num_epochs) \
        .map(lambda *x: dict(zip(header_list, x)))  # Map list to dictionary.


def input_fn_tfrecords(files_name_pattern, num_epochs, batch_size, mode):
    """
    Input functions which parses TFRecords.
    :param files_name_pattern: File name to TFRecords.
    :param num_epochs: Number of epochs.
    :param batch_size: Batch size.
    :param mode: Input function mode.
    :return: features and label.
    """
    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=files_name_pattern,
        batch_size=batch_size,
        features=get_metadata().schema.as_feature_spec(),
        reader=tf.data.TFRecordDataset,
        num_epochs=num_epochs,
        shuffle=True if mode == tf.estimator.ModeKeys.TRAIN else False,
        shuffle_buffer_size=1 + (batch_size * 2),
        prefetch_buffer_size=1,
    )


def input_fn(files_name_pattern,
             num_epochs,
             batch_size,
             mode,
             input_format,
             **kwargs):
    """
    General input function which parses csv, json, or tfrecords.
    :param files_name_pattern: File name pattern.
    :param num_epochs: Number of epochs.
    :param batch_size: Batch size.
    :param mode: Input function mode.
    :param input_format: Input type: csv or tfrecords
    :param kwargs: Other arguments to the input functions.
    :return: features and label
    """
    input_fn_reference = input_fn_csv if input_format == 'csv' else input_fn_tfrecords

    input_ref = input_fn_reference(files_name_pattern, num_epochs, batch_size, mode, **kwargs)
    iterator = input_ref.make_one_shot_iterator()
    features = iterator.get_next()

    label = features.pop(config_key('model.label', config=kwargs.get('config')))

    if mode == tf.estimator.ModeKeys.PREDICT:
        return features

    return features, label


def json_serving_input_fn(**kwargs):
    """
    Build the serving inputs.
    :param kwargs: Other keyword arguments.

    """

    def return_fn():
        inputs = {}
        label = config_key('model.label', config=kwargs.get('config'))

        for feature, value in get_metadata().schema.column_schemas.items():
            if feature != label:
                inputs[feature] = tf.placeholder(
                    shape=[None],
                    dtype=value.domain.dtype
                )

        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    return return_fn
