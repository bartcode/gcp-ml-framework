"""
Determine features for the model.
"""
import tensorflow as tf

from ..data.input import get_metadata
from ..utilities.config import config_key, config_path


def create_wide_and_deep_columns(local_config):
    """
    Create tuple of wide and deep columns respectively.
    Based on https://github.com/GoogleCloudPlatform/tf-estimator-tutorials/blob/master/00_Miscellaneous/
    tf_transform/tft-02%20-%20Babyweight%20Estimation%20with%20Transformed%20Data.ipynb
    :param local_config: Local configuration to use (config.yml).
    :return: Tuple of wide and deep columns
    """
    wide_columns = []
    deep_columns = []

    column_schemas = get_metadata(config_path('path.train-files',
                                              config=local_config,
                                              max_len=1)).schema.as_feature_spec()

    categorical_features = config_key('model.columns.categorical', config=local_config)

    label = config_key('model.label', config=local_config)
    key = config_key('model.key', config=local_config)

    for feature_name in column_schemas:
        if feature_name in [label, key]:
            continue

        # Create numerical features
        if column_schemas[feature_name].dtype == tf.float32:
            deep_columns.append(tf.feature_column.numeric_column(feature_name))

        # Create categorical features with identity
        elif column_schemas[feature_name].dtype == tf.int64:
            deep_columns.append(tf.feature_column.numeric_column(feature_name))

            if feature_name in categorical_features:
                wide_columns.append(
                    tf.feature_column.categorical_column_with_identity(
                        feature_name, num_buckets=categorical_features[feature_name])
                )

    # wide_columns += [
    #     tf.feature_column.crossed_column(['column_1', 'column_2'], hash_bucket_size=100),
    # ]

    return wide_columns, deep_columns
