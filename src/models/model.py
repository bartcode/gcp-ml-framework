"""
Create an estimator.
"""
import tensorflow as tf

from ..features.model import create_wide_and_deep_columns
from ..utilities.config import config_key


def build_estimator(config, local_config, hidden_units=None):
    """
    Build estimator. This could be any model.
    :param config: Configuration of estimator.
    :param local_config: Configuration from config.yml file in root of the project.
    :param hidden_units: Layers of the DNN.
    :return: DNNLinearCombinedRegressor.
    """
    # TODO: Add config
    wide_feature_columns, deep_feature_columns = create_wide_and_deep_columns(local_config)

    estimator = tf.estimator.DNNLinearCombinedRegressor(
        config=config,
        linear_feature_columns=wide_feature_columns,
        dnn_feature_columns=deep_feature_columns,
        dnn_hidden_units=hidden_units or [1024, 512, 256, 128, 64, 32],
        dnn_dropout=config_key('model.dropout', local_config) or .25,
    )

    # Forward key instances, such that predictions can be matched with a specific key.
    key = config_key('model.key', local_config)

    if key:
        estimator = tf.contrib.estimator.forward_features(estimator, key)

    estimator = tf.contrib.estimator.add_metrics(estimator, metric_rmse)

    return estimator


def metric_rmse(labels, predictions):
    """
    Determine Root Mean Squared Error (RMSE).
    :param labels: Labels
    :param predictions: Predictions
    :return: TF Metric
    """
    return {
        'rmse': tf.metrics.root_mean_squared_error(labels, predictions['predictions'])
    }
