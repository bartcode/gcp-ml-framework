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

from ..utilities.config import config_key, config_path


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
                project=config_key('gcp.project'), timestamp=datetime.now().strftime('%Y%m%d%H%M%S')
            )),
            staging_location=config_path('path.staging'),
            temp_location=config_path('path.temp'),
            region=config_key('gcp.region'),
            project=config_key('gcp.project'),
            zone=config_key('gcp.zone'),
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
    Partitions the data set into a train
    :param args: Nothing happens with this.
    :return: 1 or 0: whether the record will be
    """
    return int(random.uniform(0, 1) >= .8)


class DataPipeline(object):
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
    def train_test_split(training_set):
        """
        Split data set 80/20.
        :param training_set: Transformed training set
        :return: Tuple of PCollections (train, eval)
        """
        return training_set | 'TrainTestSplit' >> beam.Partition(partition_train_eval, 2)

    def execute(self):
        """
        Description of the pipeline.
        :return: None
        """
        pass

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """
        Remove context and execute pipeline.
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return: None
        """
        self._context.__exit__(None, None, None)
        self._pipeline.__exit__(None, None, None)
