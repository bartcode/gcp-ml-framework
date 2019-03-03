"""
Collects arguments to train a model on ML Engine.
"""
from .utilities.arguments import get_arguments
from .models.train_eval import train_and_evaluate


def main():
    """
    Direct execution of task to training and evaluation.
    :return: None
    """
    train_and_evaluate(get_arguments())


if __name__ == '__main__':
    main()
