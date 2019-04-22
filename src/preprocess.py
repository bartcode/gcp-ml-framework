"""
Pre-processes data using Apache Beam. This also allows for
running on Google Cloud Dataflow.
"""
import argparse
import logging
import os
import pprint
import tempfile
from datetime import datetime

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform.beam as tft_beam
from apache_beam.options.pipeline_options import StandardOptions, SetupOptions, PipelineOptions, GoogleCloudOptions, \
    DebugOptions
from tensorflow_transform.coders import CsvCoder, example_proto_coder
from tensorflow_transform.tf_metadata import dataset_schema

from .data.pipeline import get_file_info, RecommenderCombiner, RecommenderPipeline
from .features.preprocess import preprocess_recommender
from .utilities import load_config, config_path, config_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pipeline_options(config):
    """
    Get apache beam pipeline options
    :return: Dictionary
    """
    logger.info('Running Beam pipeline...')

    options = PipelineOptions(flags=[])

    if config.get('execution') == 'cloud':
        logging.info('Start running in the cloud...')

        google_cloud_options = options.view_as(GoogleCloudOptions)
        google_cloud_options.job_name = '{project}-{timestamp}'.format(
            project=config_key('cloud.project', config=config),
            timestamp=datetime.now().strftime('%Y%m%d%H%M%S')
        )
        google_cloud_options.staging_location = config_path('path.staging', config=config)
        google_cloud_options.temp_location = config_path('path.temp', config=config)
        google_cloud_options.region = config_key('cloud.region', config=config)
        google_cloud_options.project = config_key('cloud.project', config=config)
        # google_cloud_options.machine_type = ''

        options.view_as(StandardOptions).runner = 'DataflowRunner'

    setup = options.view_as(SetupOptions)
    setup.setup_file = './setup.py'
    setup.save_main_session = True

    debug = options.view_as(DebugOptions)
    # debug.dataflow_job_file = 'debug.log'
    debug.add_experiment('ignore_py3_minor_version')  # Doesn't harm when using Python 2.

    logger.info('Pipeline options:\n%s', pprint.pformat(options.get_all_options(), 4))

    return options


#
# @contextmanager
# def get_pipeline(options=None):
#     """
#     Create pipeline for Beam and automatically uses either configuration to run
#     on Dataflow or locally.
#     :return: Beam pipeline
#     """
#     pipeline_options = options if options else get_pipeline_options()
#
#     temporary_dir = pipeline_options.get_all_options()['temp_location'] \
#         if pipeline_options.get_all_options()['temp_location'] \
#         else tempfile.mkdtemp()
#
#     with beam.Pipeline(options=pipeline_options) as p:
#         with tft_beam.Context(temp_dir=temporary_dir):  # , use_deep_copy_optimization=True
#             yield p
#
#
@beam.ptransform_fn
def read_csv(pcollection, file_name, headers, metadata, key_column=None, **kwargs):
    """
    Reads a CSV file into the Beam pipeline.
    :param pcollection: Beam pcollection.
    :param file_name: Path to file.
    :param headers: Headers of file to read.
    :param metadata: Metadata of file to read.
    :param key_column: Column to use as key.
    :return: Either a PCollection (dictionaries) with a key or without a key.
    """
    logger.info('Reading CSV: %s', file_name)

    converter = CsvCoder(
        column_names=headers,
        schema=metadata.schema,
        delimiter=','
    )

    decoded_collection = pcollection \
                         | 'ReadData' >> beam.io.ReadFromText(file_name, skip_header_lines=1) \
                         | 'ParseData' >> beam.Map(converter.decode)

    if key_column:
        return decoded_collection | 'ExtractKey' >> beam.Map(lambda x: (x[key_column], x))

    return decoded_collection


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--execution',
        help='Method of execution. On the GCP or local.',
        choices=['local', 'cloud'],
        default='local'
    )

    args, _ = parser.parse_known_args()

    config = load_config('./config.yml')
    config['execution'] = args.execution

    file_out = config['cloud']['bucket'] + 'data/processed/kfi_hc'
    temp_dir = config['cloud']['bucket'] + 'temp'

    paths = {
        'transform_fn': os.path.join(config_path('path.processed', config=config), 'transform_fn'),
        'metadata': os.path.join(os.path.join(config_path('path.processed', config=config), 'transformed_metadata')),
        'processed_kfi': os.path.join(config_path('path.processed', config=config), 'keys_for_indices'),
        'processed_ifk': os.path.join(config_path('path.processed', config=config), 'indices_for_keys'),
    }

    recommender_columns = {
        'keys': config_key('model.recommender.keys', config=config),
        'indices': config_key('model.recommender.indices', config=config),
        'values': config_key('model.recommender.values', config=config)
    }

    ratings = get_file_info(os.path.join(config_path('path.raw', config=config), 'ratings.csv'), config=config)

    pipeline_options = get_pipeline_options(config)

    temporary_dir = pipeline_options.get_all_options()['temp_location'] \
        if pipeline_options.get_all_options()['temp_location'] \
        else tempfile.mkdtemp()

    logger.info('Using temporary directory: %s', temporary_dir)

    with beam.Pipeline(options=pipeline_options) as pipeline:
        with tft_beam.Context(temp_dir=temporary_dir):  # , use_deep_copy_optimization=True
            raw_data = pipeline \
                       | 'ReadData' >> read_csv(**ratings)

            # Transform
            (transformed_data, transformed_metadata), transform_fn = \
                (raw_data, ratings['metadata']) \
                | tft_beam.AnalyzeAndTransformDataset(lambda x: preprocess_recommender(x, recommender_columns))

            _ = transform_fn \
                | 'WriteTransformFn' >> tft_beam.transform_fn_io.WriteTransformFn(paths['transform_fn'])

            _ = transformed_metadata \
                | 'WriteRawMetadata' >> tft_beam.WriteMetadata(paths['metadata'], pipeline=pipeline)

            # Do a group-by to create users_for_item and items_for_user
            keys_for_indices = transformed_data \
                               | 'Group by indices' >> RecommenderPipeline.group_by_kind(key='indices', index='keys')

            indices_for_keys = transformed_data \
                               | 'Group by keys' >> RecommenderPipeline.group_by_kind(key='keys', index='indices')

            _ = keys_for_indices \
                | 'Save keys for indices (txt-debug)' >> beam.io.WriteToText(paths['processed_kfi'],
                                                                             file_name_suffix='.txt')

            _ = indices_for_keys \
                | 'Save indices for keys (txt-debug)' >> beam.io.WriteToText(paths['processed_ifk'],
                                                                             file_name_suffix='.txt')

            output_coder = example_proto_coder.ExampleProtoCoder(dataset_schema.from_feature_spec({
                'keys': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
                'indices': tf.VarLenFeature(dtype=tf.int64),
                'values': tf.VarLenFeature(dtype=tf.float32)
            }))

            _ = keys_for_indices \
                | 'Save keys for indices' >> beam.io.tfrecordio.WriteToTFRecord(paths['processed_kfi'],
                                                                                coder=output_coder,
                                                                                file_name_suffix='.tfrecords')

            _ = indices_for_keys \
                | 'Save indices for keys' >> beam.io.tfrecordio.WriteToTFRecord(paths['processed_ifk'],
                                                                                coder=output_coder,
                                                                                file_name_suffix='.tfrecords')
