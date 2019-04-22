from abc import abstractmethod

import apache_beam as beam

from src.data.pipeline import partition_train_eval


class DataPipeline:
    """
    Main class for a data pipeline.
    """

    def __init__(self, pipeline):
        """
        Initialise pipeline and context.
        """
        self.pipeline = pipeline

    @abstractmethod
    def execute(self):
        """
        Description of the pipeline.
        :return: None
        """
        pass

    @staticmethod
    def train_test_split(training_set):
        """
        Split data set 80/20.
        :param training_set: Transformed training set
        :return: Tuple of PCollections (train, eval)
        """
        return training_set | 'TrainTestSplit' >> beam.Partition(partition_train_eval, 2)
