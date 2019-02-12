"""
Collects arguments to train a model on ML Engine.
"""
import argparse
import json
import os

from .models.model import train_and_evaluate
from .utilities import load_config


def main():
    """
    Main function which handles all arguments.
    :return: None
    """
    config = load_config('./config.yml')

    # pylint: disable=C0103
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
        '--base_directory',
        help='GCS file or local config to training data',
        nargs='+',
        default=config['path']['base']
    )

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models.',
        default=os.path.join(config['path']['base'], config['path']['models'],
                             json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get('trial', ''))

    )

    parser.add_argument(
        '--staging-bucket',
        help='GCS staging location.',
        default=os.path.join(config['cloud']['bucket'], config['path']['staging'])
    )

    parser.add_argument(
        '--num-epochs',
        help="""Maximum number of training data epochs on which to train.
                If both --max-steps and --num-epochs are specified,
                the training job will run for --max-steps or --num-epochs,
                whichever occurs first. If unspecified will run for --max-steps.""",
        default=config['model']['num-epochs']
    )

    parser.add_argument(
        '--train-files',
        help='Path to training tfrecords',
        default=os.path.join(config['path']['base'], config['path']['processed'], 'train.tfrecords')
    )

    parser.add_argument(
        '--eval-files',
        help='Path to evaluation tfrecords',
        default=os.path.join(config['path']['base'], config['path']['processed'], 'eval.tfrecords')
    )

    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        default=config['model']['train-batch-size'],
        type=int
    )

    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        default=config['model']['eval-batch-size'],
        type=int
    )

    parser.add_argument(
        '--embedding-size',
        help='Number of embedding dimensions for categorical columns',
        default=config['model']['embedding-size'],
        type=int
    )

    parser.add_argument(
        '--first-layer-size',
        help='Number of nodes in the first layer of the DNN',
        default=config['model']['first-layer-size'],
        type=int
    )

    parser.add_argument(
        '--num-layers',
        help='Number of layers in the DNN',
        default=config['model']['num-layers'],
        type=int
    )

    parser.add_argument(
        '--scale-factor',
        help='How quickly should the size of the layers in the DNN decay',
        default=config['model']['scale-factor'],
        type=float
    )

    parser.add_argument(
        '--train-steps',
        help="""Steps to run the training job for. If --num-epochs is not specified,
                this must be. Otherwise the training job will run indefinitely.""",
        default=config['model']['train-steps'],
        type=int
    )

    parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evaluation for at each checkpoint',
        default=config['model']['eval-steps'],
        type=int
    )

    parser.add_argument(
        '--export-format',
        help='The input format of the exported SavedModel binary',
        choices=['JSON'],
        default=config['model']['export-format'].upper()
    )

    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default=config['model']['verbosity'].upper()
    )

    arguments = parser.parse_args()

    train_and_evaluate(arguments)


if __name__ == '__main__':
    main()
