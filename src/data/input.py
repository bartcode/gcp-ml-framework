"""
Contains methods that transform the data in such a way that it becomes
input for a model.
"""
import tensorflow as tf

from ..utilities import load_config

CONFIG = load_config('./config.yml')


# TODO: Allow different file formats than just TFRecords.
def input_fn(files_name_pattern,
             num_epochs=None,
             batch_size=200,
             mode=tf.estimator.ModeKeys.TRAIN):
    """
    Input functions which parses TFRecords.
    :param files_name_pattern: File name to TFRecords.
    :param num_epochs: Number of epochs.
    :param batch_size: Batch size.
    :param mode: Input function mode
    :return: features and label
    """
    data_set = tf.contrib.data.make_batched_features_dataset(
        file_pattern=files_name_pattern,
        batch_size=batch_size,
        features=METADATA.schema.as_feature_spec(),  # This doesn't work yet.
        reader=tf.data.TFRecordDataset,
        num_epochs=num_epochs,
        shuffle=True if mode == tf.estimator.ModeKeys.TRAIN else False,
        shuffle_buffer_size=1 + (batch_size * 2),
        prefetch_buffer_size=1,
    )

    iterator = data_set.make_one_shot_iterator()

    features = iterator.get_next()

    label = features.pop(CONFIG['model']['label'])

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
