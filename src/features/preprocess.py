"""
Methods that preprocess the data using Tensorflow Transform
"""
import tensorflow as tf
import tensorflow_transform as tft

from ..utilities.config import config_key


def preprocessor(element):
    """
    Applies TensorFlow Transform methods on vectors.
    :param element: Input vectors.
    :return: Transformed inputs.
    """
    inputs_filtered = {
        config_key('model.key'): element[config_key('model.key')],
        config_key('model.label'): element[config_key('model.label')]
    }

    for feature in element.keys():
        if feature in [config_key('model.key'), config_key('model.label')]:
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
