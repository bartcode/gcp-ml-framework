"""
Pre-processes data using Apache Beam. This also allows for
running on Google Cloud Dataflow.
"""
from .data.pipeline import DataPipeline

if __name__ == '__main__':
    PIPELINE = DataPipeline()
    PIPELINE.execute()
