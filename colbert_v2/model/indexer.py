# models/indexer.py
from colbert import Indexer
from colbert.infra import Run, RunConfig
from colbert.config import Config
from data.collection import Collection

class ColBERTIndexer:
    def __init__(self, config, collection_path):
        self.config = config
        self.collection = Collection(collection_path).load_collection()

    def index_documents(self):
        with Run().context(RunConfig(nranks=1, experiment='indexing')):
            indexer = Indexer(checkpoint=self.config.CHECKPOINT, config=self.config)
            index_name = self.config.INDEX_NAME
            indexer.index(name=index_name, collection=self.collection, overwrite=True)
        print("Index created successfully!")
