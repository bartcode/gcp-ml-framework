"""
Create an estimator.
"""
from __future__ import print_function

import tensorflow as tf

from ..models.train_eval import metric_rmse
from ..utilities.config import config_key
from ..features.model import create_wide_and_deep_columns


def build_estimator(config, hidden_units=None):
    """
    Build estimator. This could be any model.
    :param config: Configuration of estimator.
    :param hidden_units: Layers of the DNN.
    :return: DNNLinearCombinedRegressor.
    """
    wide_feature_columns, deep_feature_columns = create_wide_and_deep_columns()

    estimator = tf.estimator.DNNLinearCombinedRegressor(
        config=config,
        linear_feature_columns=wide_feature_columns,
        dnn_feature_columns=deep_feature_columns,
        dnn_hidden_units=hidden_units or [1024, 512, 256, 128, 64, 32],
        dnn_dropout=config_key('model.dropout') or .25,
    )

    # Forward key instances, such that predictions can be matched with a specific key.
    key = config_key('model.key')

    if key:
        estimator = tf.contrib.estimator.forward_features(estimator, key)

    estimator = tf.contrib.estimator.add_metrics(estimator, metric_rmse)

    return estimator
