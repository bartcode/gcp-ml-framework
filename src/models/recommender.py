"""
Relevant methods for a (WALS) recommender system.
"""
import tensorflow as tf


def find_top_k(keys, indices, top_k):
    """
    Find the top K items for the recommender system.
    :param keys: Keys
    :param indices: Indices
    :param top_k: Number of items to find.
    :return: Top K items for recommender.
    """
    items = tf.matmul(tf.expand_dims(keys, 0), tf.transpose(indices))
    top_items = tf.nn.top_k(items, k=top_k)

    return tf.cast(top_items.indices, dtype=tf.int64)
