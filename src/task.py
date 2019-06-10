"""
Collects arguments to train a model on ML Engine.
"""
import argparse

from .utilities import load_config
from .utilities.arguments import get_arguments


def main():
    """
    Direct execution of task to training and evaluation.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['recommender', 'regressor'], default='regressor')
    args, _ = parser.parse_known_args()

    if args.model == 'regressor':
        from .models.regression import train_and_evaluate
    else:
        from .models.recommender import train_and_evaluate

    train_and_evaluate(get_arguments(), load_config('./config.yml'))


if __name__ == '__main__':
    main()
