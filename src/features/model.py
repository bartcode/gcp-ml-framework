"""
Determine features for the model.
"""
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_schema

from ..data.input import get_metadata
from ..utilities.config import config_key


def create_wide_and_deep_columns():
    """
    Create tuple of wide and deep columns respectively.
    :return: Tuple of wide and deep columns
    """
    wide_columns = []
    deep_columns = []

    column_schemas = get_metadata().schema.column_schemas

    for feature_name in column_schemas:
        if feature_name in [config_key('model.label'), config_key('model.key')]:
            continue

        # Create numerical features
        if isinstance(column_schemas[feature_name].domain, dataset_schema.FloatDomain):
            deep_columns.append(tf.feature_column.numeric_column(feature_name))

        # Create categorical features with identity
        elif isinstance(column_schemas[feature_name].domain, dataset_schema.IntDomain):
            if column_schemas[feature_name].domain.is_categorical:
                wide_columns.append(
                    tf.feature_column.categorical_column_with_identity(
                        feature_name, num_buckets=column_schemas[feature_name].domain.max_value + 1)
                )
            else:
                deep_columns.append(tf.feature_column.numeric_column(feature_name))

    # wide_columns += [
    #     tf.feature_column.crossed_column(['column_1', 'column_2'], hash_bucket_size=100),
    # ]

    return wide_columns, deep_columns
