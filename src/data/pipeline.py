"""
Define pipeline methods.
"""
import logging
import os
import random
import tempfile
from datetime import datetime

import apache_beam as beam
import tensorflow_transform.beam as tft_beam

from ..utilities import load_config

CONFIG = load_config('./config.yml')


def get_pipeline_options():
    """
    Get apache beam pipeline options
    :return: Dictionary
    """
    logging.info('Running Beam pipeline...')

    options = dict()

    if os.environ.get('EXECUTOR') == 'cloud':
        logging.info('Start running in the cloud...')

        options = dict(
            runner='DataflowRunner',
            job_name=('{project}-{timestamp}'.format(
                project=CONFIG['gcp']['project'], timestamp=datetime.now().strftime('%Y%m%d%H%M%S')
            )),
            staging_location=os.path.join(CONFIG['path']['base'], CONFIG['path']['staging']),
            temp_location=os.path.join(CONFIG['path']['base'], CONFIG['path']['temp']),
            region=CONFIG['gcp']['region'],
            project=CONFIG['gcp']['project'],
            zone=CONFIG['gcp']['zone'],
            autoscaling_algorithm='THROUGHPUT_BASED',
            save_main_session=True,
            setup_file='./setup.py'
        )

    return options


def debug(element):
    """
    Can be used in beam.Map() to see the current element in the pipeline (only when executed locally).
    :param element: Element in pipeline to debug
    :return: Element
    """
    if os.environ.get('EXECUTOR') != 'cloud':
        import pdb
        pdb.set_trace()

    return element


def partition_train_eval(*args):
    """
    Partitions the dataset into a train
    :param args: Nothing happens with this.
    :return: 1 or 0: whether the record will be
    """
    return int(random.uniform(0, 1) >= .8)


class DataPipeline:
    """
    Main class for a data pipeline.
    """

    def __init__(self):
        """
        Initialise pipeline and context.
        """
        pipeline_options = get_pipeline_options()

        self._pipeline = beam.Pipeline(options=beam.pipeline.PipelineOptions(flags=[], **pipeline_options))

        temporary_dir = pipeline_options['temp_location'] \
            if 'temp_location' in pipeline_options \
            else tempfile.mkdtemp()

        self._context = tft_beam.Context(temp_dir=temporary_dir)

    @staticmethod
    def train_test_split(training):
        """
        Split dataset 80/20.
        :param training: Transformed training set
        :return: Tuple of PCollections (train, eval)
        """
        return training | 'TrainTestSplit' >> beam.Partition(partition_train_eval, 2)

    def execute(self):
        """
        Description of the pipeline.
        :return: None
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Remove context and execute pipeline.
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return: None
        """
        self._context.__exit__()
        self._pipeline.__exit__()
