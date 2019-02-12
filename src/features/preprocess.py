"""
Methods that preprocess the data using Tensorflow Transform
"""
import tensorflow as tf
import tensorflow_transform as tft

from ..utilities import load_config

CONFIG = load_config('./config.yml')


def preprocessor(element):
    """
    Applies TensorFlow Transform methods on vectors.
    :param element: Input vectors.
    :return: Transformed inputs.
    """
    inputs_filtered = {
        CONFIG['model']['key']: element[CONFIG['model']['key']],
        CONFIG['model']['label']: element[CONFIG['model']['label']]
    }

    for feature in element.keys():
        if feature == 'card_id' or feature == CONFIG['model']['label']:
            continue

        if element[feature].dtype == tf.string:
            inputs_filtered[feature] = tft.compute_and_apply_vocabulary(element[feature])
        else:
            if '_sum' in feature:
                feature_basis = feature[:-len('_sum')]
                inputs_filtered[feature_basis + '_avg'] = element[feature_basis + '_sum'] / element['transactions']

            inputs_filtered[feature + '_0_1'] = tft.scale_to_0_1(element[feature])
            inputs_filtered[feature + '_z'] = tft.scale_to_z_score(element[feature])

    return inputs_filtered
