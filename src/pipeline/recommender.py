import apache_beam as beam
import tensorflow as tf
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_schema

from src.utilities import config_key
from ..data.pipeline import read_csv
from ..features.preprocess import preprocess_recommender
from ..pipeline.default import DataPipeline


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


class RecommenderPipeline(DataPipeline, object):
    """
    Example of a pipeline for a recommender system.
    """
    RECOMMENDER_COLUMNS = {}

    def __init__(self, pipeline, **kwargs):
        super(RecommenderPipeline, self).__init__(pipeline)

        self.input = kwargs.get('input')
        self.output= kwargs.get('output')
        self.config = kwargs.get('config', {})

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

    def execute(self):
        """
        Starts the data pipeline.
        :return: None
        """
        recommender_columns = {
            'keys': config_key('model.recommender.keys', config=self.config),
            'indices': config_key('model.recommender.indices', config=self.config),
            'values': config_key('model.recommender.values', config=self.config)
        }

        raw_data = self.pipeline | 'ReadData' >> read_csv(**self.input['ratings'])

        # Transform
        (transformed_data, transformed_metadata), transform_fn = \
            (raw_data, self.input['ratings']['metadata']) \
            | tft_beam.AnalyzeAndTransformDataset(lambda x: preprocess_recommender(x, recommender_columns))

        _ = transform_fn \
            | 'WriteTransformFn' >> tft_beam.transform_fn_io.WriteTransformFn(self.output['transform_fn'])

        _ = transformed_metadata \
            | 'WriteRawMetadata' >> tft_beam.WriteMetadata(self.output['metadata'], pipeline=self.pipeline)

        # Do a group-by to create users_for_item and items_for_user
        keys_for_indices = transformed_data \
                           | 'Group by indices' >> RecommenderPipeline.group_by_kind(key='indices', index='keys')

        indices_for_keys = transformed_data \
                           | 'Group by keys' >> RecommenderPipeline.group_by_kind(key='keys', index='indices')

        _ = keys_for_indices \
            | 'Save keys for indices (txt-debug)' >> beam.io.WriteToText(self.output['processed_kfi'],
                                                                         file_name_suffix='.txt')

        _ = indices_for_keys \
            | 'Save indices for keys (txt-debug)' >> beam.io.WriteToText(self.output['processed_ifk'],
                                                                         file_name_suffix='.txt')

        output_coder = example_proto_coder.ExampleProtoCoder(dataset_schema.from_feature_spec({
            'keys': tf.FixedLenFeature(shape=[1], dtype=tf.int64),
            'indices': tf.VarLenFeature(dtype=tf.int64),
            'values': tf.VarLenFeature(dtype=tf.float32)
        }))

        _ = keys_for_indices \
            | 'Save keys for indices' >> beam.io.tfrecordio.WriteToTFRecord(self.paths['processed_kfi'],
                                                                            coder=output_coder,
                                                                            file_name_suffix='.tfrecords')

        _ = indices_for_keys \
            | 'Save indices for keys' >> beam.io.tfrecordio.WriteToTFRecord(self.paths['processed_ifk'],
                                                                            coder=output_coder,
                                                                            file_name_suffix='.tfrecords')
