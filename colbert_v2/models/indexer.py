# models/indexer.py
from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from config import Config
import os 

class ColBERTIndexer:
    def __init__(self, config, collection_path):
        self.config = config
        queries = os.path(config.QUERIES_PATH)
        collection = os.path(config.COLLECTION_PATH)
        self.queries = Queries(path=queries)
        self.collection = Collection(path=collection)
    

    def index_documents(self):
        with Run().context(RunConfig(nranks=1, experiment='indexing')):
            config = ColBERTConfig(doc_maxlen=512, nbits=2, kmeans_niters=8)
            indexer = Indexer(checkpoint=self.config.CHECKPOINT, config=config)
            indexer.index(name=self.config.INDEX_NAME, collection=self.collection, overwrite=True)
        print("Index created successfully!")
