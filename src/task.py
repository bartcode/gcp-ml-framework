"""
Collects arguments to train a model on ML Engine.
"""
from .models.regression import train_and_evaluate
from .utilities import load_config
from .utilities.arguments import get_arguments


def main():
    """
    Direct execution of task to training and evaluation.
    :return: None
    """
    train_and_evaluate(get_arguments(), load_config('./config.yml'))


if __name__ == '__main__':
    main()
