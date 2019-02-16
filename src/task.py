"""
Collects arguments to train a model on ML Engine.
"""
import argparse
import json
import os

from .models.model import train_and_evaluate
from .utilities import config_path, config_key


def main():
    """
    Main function which handles all arguments.
    :return: None
    """
    # pylint: disable=C0103
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--base_directory',
        help='GCS file or local config to training data',
        nargs='+',
        default=config_key('path.base')
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models.',
        default=os.path.join(config_path('path.models'),
                             json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', ''))

    )

    parser.add_argument(
        '--staging-bucket',
        help='GCS staging location.',
        default=os.path.join(config_key('cloud.bucket'), config_key('path.staging'))
    )

    parser.add_argument(
        '--num-epochs',
        help=('Maximum number of training data epochs on which to train.\n'
              'If both --max-steps and --num-epochs are specified,\n'
              'the training job will run for --max-steps or --num-epochs,\n'
              'whichever occurs first. If unspecified will run for --max-steps.'),
        default=config_key('model.num-epochs')
    )

    parser.add_argument(
        '--train-files',
        help='Path to training file',
        default=os.path.join(config_path('path.processed', 'path.train-files'))
    )

    parser.add_argument(
        '--eval-files',
        help='Path to evaluation file',
        default=os.path.join(config_path('path.processed', 'path.eval-files'))
    )

    parser.add_argument(
        '--test-files',
        help='Path to test file',
        default=os.path.join(config_path('path.processed', 'path.test-files'))
    )

    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        default=config_key('model.train-batch-size'),
        type=int
    )

    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        default=config_key('model.eval-batch-size'),
        type=int
    )

    parser.add_argument(
        '--embedding-size',
        help='Number of embedding dimensions for categorical columns',
        default=config_key('model.embedding-size'),
        type=int
    )

    parser.add_argument(
        '--first-layer-size',
        help='Number of nodes in the first layer of the DNN',
        default=config_key('model.first-layer-size'),
        type=int
    )

    parser.add_argument(
        '--num-layers',
        help='Number of layers in the DNN',
        default=config_key('model.num-layers'),
        type=int
    )

    parser.add_argument(
        '--scale-factor',
        help='How quickly should the size of the layers in the DNN decay',
        default=config_key('model.scale-factor'),
        type=float
    )

    parser.add_argument(
        '--train-steps',
        help="""Steps to run the training job for. If --num-epochs is not specified,
                this must be. Otherwise the training job will run indefinitely.""",
        default=config_key('model.train-steps'),
        type=int
    )

    parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evaluation for at each checkpoint',
        default=config_key('model.eval-steps'),
        type=int
    )

    parser.add_argument(
        '--input-format',
        help='The input format of the train and evaluation sets.',
        choices=['csv', 'tfrecords'],
        default=config_key('model.input-format').upper()
    )

    parser.add_argument(
        '--export-format',
        help='The format of the exported SavedModel binary',
        choices=['JSON'],
        default=config_key('model.export-format').upper()
    )

    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default=config_key('model.verbosity').upper()
    )

    train_and_evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
