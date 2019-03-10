"""
Define pipeline methods.
"""
import logging
import os
import random
import tempfile
from abc import abstractmethod
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import apache_beam as beam
from apache_beam import io
import tensorflow as tf
from apache_beam.pvalue import PCollection
from tensorflow_transform import coders
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata.dataset_metadata import DatasetMetadata

from ..features.preprocess import preprocess_recommender
from ..data.input import get_metadata, get_headers
from ..utilities.config import config_key, config_path, cloud_execution


def get_pipeline_options() -> Dict[str, Any]:
    """
    Get apache beam pipeline options
    :return: Dictionary
    """
    logging.info('Running Beam pipeline...')

    options = dict()

    if cloud_execution():
        logging.info('Start running in the cloud...')

        options = dict(
            runner='DataflowRunner',
            job_name=('{project}-{timestamp}'.format(
                project=config_key('cloud.project'), timestamp=datetime.now().strftime('%Y%m%d%H%M%S')
            )),
            staging_location=config_path('path.staging'),
            temp_location=config_path('path.temp'),
            region=config_key('cloud.region'),
            project=config_key('cloud.project'),
            zone=config_key('cloud.zone'),
            autoscaling_algorithm='THROUGHPUT_BASED',
            save_main_session=True,
            setup_file='./setup.py'
        )

    return options


def pdebug(element: Any) -> Any:
    """
    Can be used in beam.Map() to see the current element in the pipeline (only when executed locally).
    :param element: Element in pipeline to debug
    :return: Element
    """
    if not cloud_execution():
        import pdb
        pdb.set_trace()

    return element


def partition_train_eval(*args):
    """
    Partitions the data set into a train
    :param args: Nothing happens with this.
    :return: 1 or 0: whether the record will be
    """
    # pylint: disable=unused-argument
    return int(random.uniform(0, 1) >= .8)


@contextmanager
def get_pipeline():
    """
    Create pipeline for Beam and automatically uses either configuration to run
    on Dataflow or locally.
    :return: Beam pipeline
    """
    pipeline_options = get_pipeline_options()

    temporary_dir = pipeline_options['temp_location'] \
        if 'temp_location' in pipeline_options \
        else tempfile.mkdtemp()

    with beam.Pipeline(options=beam.pipeline.PipelineOptions(flags=[], **pipeline_options)) as pipeline:
        with tft_beam.Context(temp_dir=temporary_dir):
            yield pipeline


@beam.ptransform_fn
def read_csv(pcollection: PCollection, file_name: str, key_column: Optional[str] = None,
             metadata: Optional[DatasetMetadata] = None) -> PCollection:
    """
    Reads a CSV file into the Beam pipeline.
    :param pcollection: Beam pcollection.
    :param file_name: Path to file.
    :param key_column: Column to use as key.
    :param metadata: (Customer) metadata.
    :return: Either a PCollection (dictionaries) with a key or without a key.
    """
    logging.info('Reading CSV file: %s', file_name)

    metadata = get_metadata() \
        if metadata is None \
        else metadata

    converter = coders.CsvCoder(
        column_names=get_headers(file_name),
        schema=metadata.schema,
        delimiter=','
    )

    decoded_collection = pcollection \
                         | 'ReadData' >> beam.io.ReadFromText(file_name, skip_header_lines=1) \
                         | 'ParseData' >> beam.Map(converter.decode)

    if key_column:
        return decoded_collection | 'ExtractKey' >> beam.Map(lambda x: (x[key_column], x))

    return decoded_collection


class DataPipeline(object):
    """
    Main class for a data pipeline.
    """

    def __init__(self):
        """
        Initialise pipeline and context.
        """
        pass

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


class RecommenderPipeline(DataPipeline):
    """
    Example of a pipeline for a recommender system.
    Refer to: https://towardsdatascience.com/how-to-build-a-collaborative-filtering-
                model-for-personalized-recommendations-using-tensorflow-and-b9a77dc1320
    """

    @staticmethod
    @beam.ptransform_fn
    def group_by_kind(pcollection: PCollection, key: str) -> PCollection:
        """
        Groups the PCollection by the given key.
        :param pcollection: PCollection.
        :param key: String of key to group by.
        :return: Reformatted records.
        """

        def reformat_record(element: Tuple[Any, Any], index_column: str) -> Dict[str, list]:
            """
            Reformat records such that each element contains a list of values.
            :param element: Key-value tuple.
            :param index_column: Column to use as key.
            :return: Record with three columns: keys, indices, values.
            """
            (key_name, value_list) = element

            return {
                'keys': [key_name],
                'indices': [v[index_column] for v in value_list],
                'values': [v['values'] for v in value_list]
            }

        return pcollection \
               | 'Create key-value pair' >> beam.Map(lambda x: (x[key], x)) \
               | 'Group items' >> beam.GroupByKey() \
               | 'Reformat records' >> beam.Map(reformat_record, index_column=key)

    def execute(self):
        """
        Starts the data pipeline.
        :return: None
        """
        with get_pipeline() as pipeline:
            self.collect_data(pipeline)

    def collect_data(self, pipeline: PCollection):
        """
        Executes recommender pipeline.
        :return: None
        """
        ratings = os.path.join(config_path('path.raw'), 'ratings.csv')

        # Load metadata
        metadata = get_metadata(ratings)

        # pylint: disable=E1120
        raw_data = pipeline | 'Read ratings' >> read_csv(file_name=ratings, metadata=metadata)

        # Transform
        (transformed_data, transformed_metadata), transform_fn = \
            (raw_data, metadata) \
            | tft_beam.AnalyzeAndTransformDataset(preprocess_recommender)

        _ = transform_fn \
            | 'WriteTransformFn' >> transform_fn_io.WriteTransformFn(os.path.join(config_path('path.processed'),
                                                                                  'transform_fn'))

        _ = transformed_metadata \
            | 'WriteRawMetadata' >> tft_beam.WriteMetadata(os.path.join(os.path.join(config_path('path.processed'),
                                                                                     'transformed_metadata')), pipeline)

        # do a group-by to create users_for_item and items_for_user
        keys_for_indices = transformed_data | 'Group by indices' >> self.group_by_kind(key='indices')
        indices_for_keys = transformed_data | 'Group by keys' >> self.group_by_kind(key='keys')

        output_coder = example_proto_coder.ExampleProtoCoder(dataset_schema.from_feature_spec({
            'keys': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
            'indices': tf.VarLenFeature(dtype=tf.int64),
            'values': tf.VarLenFeature(dtype=tf.float32)
        }))

        _ = keys_for_indices \
            | 'Save keys for indices' >> io.tfrecordio.WriteToTFRecord(os.path.join(config_path('path.processed'),
                                                                                    'keys_for_indices'),
                                                                       coder=output_coder,
                                                                       file_name_suffix='.tfrecords')

        _ = indices_for_keys \
            | 'Save indices for keys' >> io.tfrecordio.WriteToTFRecord(os.path.join(config_path('path.processed'),
                                                                                    'indices_for_key'),
                                                                       coder=output_coder,
                                                                       file_name_suffix='.tfrecords')
