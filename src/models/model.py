"""
Create an estimator.
"""
from __future__ import print_function

import json
import os

import tensorflow as tf

from ..utilities.config import config_key
from ..features.model import create_wide_and_deep_columns
from ..data.input import input_fn, json_serving_input_fn


def get_session_config():
    """
    Determines tf.ConfigProto from environment variables.
    :return: Tensorflow configuration (tf.ConfigProto).
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


def build_estimator(config, hidden_units=None):
    """
    Build estimator
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
        dnn_dropout=.4,
    )

    # Forward key instances, such that predictions can be matched with a specific key.
    for key in config_key('model.key'):
        estimator = tf.contrib.estimator.forward_features(estimator, key)

    estimator = tf.contrib.estimator.add_metrics(estimator, metric_rmse)

    return estimator


def train_and_evaluate(args):
    """
    Run training and evaluate.
    :param args: Arguments for model
    :return: train_and_evaluate
    """
    train_spec = tf.estimator.TrainSpec(
        lambda: input_fn(args.train_files,
                         num_epochs=args.num_epochs,
                         batch_size=args.train_batch_size),
        max_steps=args.train_steps
    )

    exporter = tf.estimator.FinalExporter(config_key('model.name'), json_serving_input_fn)

    eval_spec = tf.estimator.EvalSpec(
        lambda: input_fn(args.eval_files,
                         mode=tf.estimator.ModeKeys.EVAL),
        steps=args.eval_steps,
        exporters=[exporter],
        name='eval',
        throttle_secs=30
    )

    run_config = tf.estimator.RunConfig(
        session_config=get_session_config(),
        save_checkpoints_secs=30,
        save_summary_steps=10000,
        keep_checkpoint_max=5
    ).replace(model_dir=args.job_dir)

    estimator = build_estimator(
        config=run_config,
        hidden_units=[
            max(2, int(args.first_layer_size * args.scale_factor ** i))
            for i in range(args.num_layers)
        ]
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
