"""
Pre-processes data using Apache Beam. This also allows for
running on Google Cloud Dataflow.
"""
import argparse
import logging
import os

from src.pipeline.recommender import RecommenderPipeline
from .data.pipeline import get_file_info, get_pipeline
from .utilities import load_config, config_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    ratings = get_file_info(os.path.join(config_path('path.raw', config=config), 'ratings.csv'), config=config)

    with get_pipeline(config) as pipeline:
        recommender = RecommenderPipeline(pipeline, ratings=ratings, paths=paths, config=config)
        recommender.execute()
