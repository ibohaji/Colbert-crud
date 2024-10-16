# models/indexer.py
import argparse
from custom.execution_monitor import ExecutionMonitor
from colbert import Indexer
from colbert.infra import ColBERTConfig, Run, RunConfig
from time import time
from ..config import Config, MetaData


class ColBERTIndexer:
    def __init__(self, config, collection_path):
        self.config = config
        #  FileProcessor.add_missing_headers(config.COLLECTION_PATH)
        #  FileProcessor.ensure_proper_header(config.COLLECTION_PATH)
        #  self.queries = Queries(path=config.QUERIES_PATH)
        self.collection_path = collection_path
   
    @ExecutionMonitor
    def index_documents(self):

        with Run().context(RunConfig(nranks=1, experiment='experiments')):
            start_time = time()
            config = ColBERTConfig(doc_maxlen=512, nbits=2)
            indexer = Indexer(checkpoint=self.config.CHECKPOINT, config=config)
            indexer.index(name=self.config.INDEX_NAME, collection=self.collection_path, overwrite=True)
        total_time = time() - start_time


        print(f"Indexing completed successfully in {total_time} seconds")
        print("Index created successfully!")
        MetaData().update(Index_time=total_time)

if __name__ == "__main__":
    custom_config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', type=str, default=custom_config.COLLECTION_PATH)
    parser.add_argument('--experiment', type=str, default="experiments")
    parser.add_argument('--index_name', type=str, default=custom_config.INDEX_NAME)
    parser.add_argument('--checkpoint', type=str, default=custom_config.CHECKPOINT)

    args = parser.parse_args()
    MetaData().update(EXPERIMENT_ID=args.experiment)
    collection_path = args.collection
    indexer = ColBERTIndexer(custom_config, collection_path = collection_path)
    indexer.index_documents()
