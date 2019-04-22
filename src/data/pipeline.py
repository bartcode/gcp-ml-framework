"""
Define pipeline methods.
"""
import logging
import pprint
import random
import tempfile
from contextlib import contextmanager
from datetime import datetime

import apache_beam as beam
import tensorflow_transform.beam as tft_beam
from apache_beam.options.pipeline_options import GoogleCloudOptions, StandardOptions, PipelineOptions, SetupOptions, \
    DebugOptions
from tensorflow_transform.coders import CsvCoder

from .input import get_headers, get_metadata
from ..utilities.config import config_key, config_path

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


@contextmanager
def get_pipeline(config):
    """
    Create pipeline for Beam and automatically uses either configuration to run
    on Dataflow or locally.
    :param config: Configuration to use.
    :return: Beam pipeline
    """
    pipeline_options = get_pipeline_options(config)

    temporary_dir = pipeline_options.get_all_options()['temp_location'] \
        if pipeline_options.get_all_options()['temp_location'] \
        else tempfile.mkdtemp()

    logger.info('Using temporary directory: %s', temporary_dir)

    temporary_dir = pipeline_options.get_all_options()['temp_location'] \
        if pipeline_options.get_all_options()['temp_location'] \
        else tempfile.mkdtemp()

    with beam.Pipeline(options=pipeline_options) as p:
        with tft_beam.Context(temp_dir=temporary_dir):  # , use_deep_copy_optimization=True
            yield p


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
