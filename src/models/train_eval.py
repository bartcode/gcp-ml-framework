"""
Training and evaluation methods.
"""
import argparse
import os
from typing import Any, Tuple

import tensorflow as tf

from ..data.input import input_fn, json_serving_input_fn
from ..models.model import build_estimator
from ..utilities import config_key
from ..utilities.config import get_tensorflow_session_config


def train_and_evaluate(args: argparse.Namespace) -> Tuple[Any, Any]:
    """
    Run training and evaluate.
    :param args: Arguments for model
    :return: train_and_evaluate
    """
    tf.logging.set_verbosity(args.verbosity)

    train_spec = tf.estimator.TrainSpec(
        lambda: input_fn(args.train_files,
                         num_epochs=args.num_epochs,
                         batch_size=args.train_batch_size,
                         mode=tf.estimator.ModeKeys.TRAIN,
                         input_format=args.input_format),
        max_steps=args.train_steps
    )

    exporter = tf.estimator.FinalExporter(config_key('model.name'), json_serving_input_fn)

    eval_spec = tf.estimator.EvalSpec(
        lambda: input_fn(args.eval_files,
                         num_epochs=args.num_epochs,
                         batch_size=args.eval_batch_size,
                         mode=tf.estimator.ModeKeys.EVAL,
                         input_format=args.input_format),
        steps=args.eval_steps,
        exporters=[exporter],
        name='eval',
        throttle_secs=30  # Seconds until next evaluation.
    )

    run_config = tf.estimator.RunConfig(
        session_config=get_tensorflow_session_config(),
        save_checkpoints_secs=30,
        save_summary_steps=5000,
        keep_checkpoint_max=5
    ).replace(model_dir=os.path.join(args.job_dir, config_key('model.name')))

    estimator = build_estimator(
        config=run_config,
        hidden_units=[
            max(2, int(args.first_layer_size * args.scale_factor ** i))
            for i in range(args.num_layers)
        ]
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
