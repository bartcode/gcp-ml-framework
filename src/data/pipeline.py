"""
Define pipeline methods.
"""
import logging
import os
import pprint
import random
import tempfile
from abc import abstractmethod
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform.beam as tft_beam
from apache_beam import io
from apache_beam.options.pipeline_options import GoogleCloudOptions, StandardOptions, PipelineOptions, SetupOptions, \
    DebugOptions
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_schema

from .input import get_headers, get_metadata
from ..features.preprocess import preprocess_recommender
from ..utilities.config import config_key, config_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pdebug(element):
    """
    Can be used in beam.Map() to see the current element in the pipeline (only when executed locally).
    :param element: Element in pipeline to debug
    :return: Element
    """
    import pdb
    pdb.set_trace()

    return element


def get_file_info(file_name, **kwargs):
    """
    Retrieves information of a file, which will be processed in the pipeline.
    :param file_name: Path to file.
    :param kwargs: Other keyword arguments.
    :return: Dictionary with headers and metadata of specified file.
    """
    file_info = {
        'file_name': file_name,
        'headers': get_headers(file_name, config=kwargs.get('config')),
        'metadata': get_metadata(file_name, config=kwargs.get('config'))
    }

    logger.info('Got file info for %s:\n%s', file_name, pprint.pformat(file_info, 4))

    return file_info


def partition_train_eval(*args):
    """
    Partitions the data set into a train
    :param args: Nothing happens with this.
    :return: 1 or 0: whether the record will be
    """
    # pylint: disable=unused-argument
    return int(random.uniform(0, 1) >= .8)


@beam.ptransform_fn
def read_csv(pcollection, file_name, headers, key_column=None, **kwargs):
    """
    Reads a CSV file into the Beam pipeline.
    :param pcollection: Beam pcollection.
    :param file_name: Path to file.
    :param headers: Headers of file to read.
    :param key_column: Column to use as key.
    :return: Either a PCollection (dictionaries) with a key or without a key.
    """
    logger.info('Reading CSV: %s', file_name)

    decoded_collection = pcollection \
                         | 'ReadData' >> beam.io.ReadFromText(file_name, skip_header_lines=1) \
                         | 'Split' >> beam.Map(lambda x: dict(zip(headers, x.split(','))))

    if key_column:
        return decoded_collection | 'ExtractKey' >> beam.Map(lambda x: (x[key_column], x))

    return decoded_collection


class DataPipeline:
    """
    Main class for a data pipeline.
    """

    def __init__(self, config):
        """
        Initialise pipeline and context.
        """
        self.config = config

    @abstractmethod
    def execute(self, pipeline):
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


class RecommenderCombiner(beam.CombineFn):
    """
    Combine elements of the recommender system.
    """

    def to_runner_api_parameter(self, unused_context):
        pass

    def add_input(self, accumulator, element, *args, **kwargs):
        """

        :param accumulator: Accumulator
        :param element: Element to add
        :param args:
        :param kwargs:
        :return:
        """
        if not accumulator['keys']:
            accumulator['keys'] = [element[kwargs['key']][0]] \
                if isinstance(element[kwargs['key']], list) else [element[kwargs['key']]]

        accumulator['indices'] += element[kwargs['index']] \
            if isinstance(element[kwargs['index']], list) else [element[kwargs['index']]]

        accumulator['values'] += element['values'] \
            if isinstance(element['values'], list) else [element['values']]

        return accumulator

    def create_accumulator(self, *args, **kwargs):
        """
        Create accumulator
        :param args: None
        :param kwargs: None
        :return: Empty accumulator dictionary.
        """
        return {
            'keys': [],
            'indices': [],
            'values': []
        }

    def merge_accumulators(self, accumulators, *args, **kwargs):
        """
        Merges multiple accumulators
        :param accumulators: List of accumulators
        :param args:
        :param kwargs:
        :return:
        """
        accumulated = self.create_accumulator(*args, **kwargs)

        for accumulator in accumulators:
            accumulated = self.add_input(accumulated, accumulator, key='keys', index='indices')

        return accumulated

    def extract_output(self, accumulator, *args, **kwargs):
        """
        Extracts output from accumulator
        :param accumulator: Accumulator
        :param args:
        :param kwargs:
        :return: Accumulator
        """
        return accumulator


class RecommenderPipeline(DataPipeline):
    """
    Example of a pipeline for a recommender system.
    Refer to: https://towardsdatascience.com/how-to-build-a-collaborative-filtering-
                model-for-personalized-recommendations-using-tensorflow-and-b9a77dc1320
    """
    RECOMMENDER_COLUMNS = {}

    @staticmethod
    @beam.ptransform_fn
    def group_by_kind(pcollection, key, index):
        """
        Groups the PCollection by the given key.
        :param pcollection: PCollection.
        :param key: String of key to group by.
        :param index: String of key to use for index.
        :return: Reformatted records.
        """
        return pcollection \
               | 'Create key-value pair' >> beam.Map(lambda x: (x[key], x)) \
               | 'CombineElements' >> beam.CombinePerKey(RecommenderCombiner(), key=key, index=index) \
               | 'Retrieve values' >> beam.Map(lambda x: x[1])

    def rename_columns(self, element):
        """
        Standardise names of columns for recommender system.
        :param element:
        :return:
        """
        return {
            'keys': element[self.RECOMMENDER_COLUMNS['keys']],
            'indices': element[self.RECOMMENDER_COLUMNS['indices']],
            'values': element[self.RECOMMENDER_COLUMNS['values']]
        }

    def execute(self, pipeline):
        """
        Starts the data pipeline.
        :return: None
        """
        pass
        # ratings = get_file_info(os.path.join(config_path('path.raw'), 'ratings.csv'))

        # ratings = {
        #     'file_name': 'gs://gcp-ml-framework/data/raw/ratings.csv',
        #     'headers': ['userId', 'movieId', 'rating', 'timestamp']
        # }

        # paths = {
        #     'transform_fn': os.path.join(config_path('path.processed'), 'transform_fn'),
        #     'metadata': os.path.join(os.path.join(config_path('path.processed'), 'transformed_metadata')),
        #     'processed_kfi': os.path.join(config_path('path.processed'), 'keys_for_indices'),
        #     'processed_ifk': os.path.join(config_path('path.processed'), 'indices_for_keys'),
        # }

        # RECOMMENDER_COLUMNS = {
        #     'keys': config_key('model.recommender.keys'),
        #     'indices': config_key('model.recommender.indices'),
        #     'values': config_key('model.recommender.values')
        # }

        # logger.info('Writing to the following paths:\n%s', pprint.pformat(paths, 4))

        # raw_data = pipeline \
        #            | 'ReadData' >> beam.io.ReadFromText(ratings['file_name'], skip_header_lines=1) \
        #            | 'Split' >> beam.Map(lambda x: x.split(',')) \
        #            | 'Save keys for indices txt' >> io.WriteToText(paths['processed_kfi'])
        # | 'Read ratings' >> read_csv(**ratings)

        # Transform
        # (transformed_data, transformed_metadata), transform_fn = \
        #     (raw_data, ratings['metadata']) \
        #     | tft_beam.AnalyzeAndTransformDataset(preprocess_recommender)

        # transformed_data = raw_data

        # _ = transform_fn \
        #     | 'WriteTransformFn' >> transform_fn_io.WriteTransformFn(paths['transform_fn'])

        # _ = transformed_metadata \
        #     | 'WriteRawMetadata' >> tft_beam.WriteMetadata(paths['metadata'], pipeline=pipeline)

        # # Do a group-by to create users_for_item and items_for_user
        # keys_for_indices = transformed_data # \
        #                    # | 'Create key-value pair' >> beam.Map(lambda x: (x['keys'], x))
        #                    # | 'Group by indices' >> RecommenderPipeline.group_by_kind(key='indices', index='keys')

        # indices_for_keys = transformed_data \
        #                    | 'Group by keys' >> RecommenderPipeline.group_by_kind(key='keys', index='indices')

        # _ = keys_for_indices \
        #     | 'Save keys for indices txt' >> io.WriteToText(paths['processed_kfi'], file_name_suffix='.txt')

        # _ = indices_for_keys \
        #     | 'Save indices for keys (txt)' >> io.WriteToText(paths['processed_ifk'], file_name_suffix='.txt')

        # output_coder = example_proto_coder.ExampleProtoCoder(dataset_schema.from_feature_spec({
        #     'keys': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
        #     'indices': tf.VarLenFeature(dtype=tf.int64),
        #     'values': tf.VarLenFeature(dtype=tf.float32)
        # }))

        # _ = keys_for_indices \
        #     | 'Save keys for indices' >> io.tfrecordio.WriteToTFRecord(paths['processed_kfi'],
        #                                                                coder=output_coder,
        #                                                                file_name_suffix='.tfrecords')
        #
        # _ = indices_for_keys \
        #     | 'Save indices for keys' >> io.tfrecordio.WriteToTFRecord(paths['processed_ifk'],
        #                                                                coder=output_coder,
        #                                                                file_name_suffix='.tfrecords')
