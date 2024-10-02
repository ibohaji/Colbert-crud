# models/indexer.py
from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from config import Config
import os 
from data.collection import FileProcessor

class ColBERTIndexer:
    def __init__(self, config, collection_path):
        self.config = config
        # FileProcessor.add_missing_headers(config.COLLECTION_PATH)
       #  FileProcessor.ensure_proper_header(config.COLLECTION_PATH)
        #self.queries = Queries(path=config.QUERIES_PATH)
        self.collection_path = collection_path

    def index_documents(self):
        with Run().context(RunConfig(nranks=2, experiment='experiments')):
            config = ColBERTConfig(doc_maxlen=512, nbits=2)
            indexer = Indexer(checkpoint=self.config.CHECKPOINT, config=config)
            indexer.index(name=self.config.INDEX_NAME, collection=self.collection_path, overwrite=True)
        print("Index created successfully!")


if __name__ == "__main__":
    custom_config = Config()
    indexer = ColBERTIndexer(custom_config, custom_config.COLLECTION_PATH)
    indexer.index_documents()