# models/searcher.py
from colbert import Searcher
from colbert.infra import (
Run,
RunConfig,
ColBERTConfig
)
from config import Config
import argparse
from colbert.data import Queries

class ColBERTSearcher:
    def __init__(self, index_name):
        self.config = Config()

    def search(self):
        with Run().context(RunConfig(nranks=2, experiment='experiments')):
            config = ColBERTConfig(root="experiments")
            searcher = Searcher(index=self.config.INDEX_NAME, config=config)
            queries = Queries(self.config.QUERIES_PATH)
            ranking = searcher.search_all(queries, k=100)  
            ranking.save('scifact.nbit=2.ranking.tsv')

if __name__ == "__main__":
    ## parse args and call searcher
    config = Config()
    searcher = ColBERTSearcher(config.INDEX_NAME)
    searcher.search()
