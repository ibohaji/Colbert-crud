# models/searcher.py
from colbert import Searcher
from colbert.infra import Run, RunConfig
from config import Config
import argparse

class ColBERTSearcher:
    def __init__(self, index_name):
        self.config = Config()

    def search(self, query):
        with Run().context(RunConfig(experiment='searching')):
            searcher = Searcher(index=self.config.INDEX_NAME, config=self.config)
            results = searcher.search(query, k=20)  

        for passage_id, passage_rank, passage_score in zip(*results):
            print(f"[{passage_rank}] {passage_score:.1f} - {searcher.collection[passage_id]}")

if __name__ == "__main__":
    ## parse args and call searcher
    config = Config()
    searcher = ColBERTSearcher(config.INDEX_NAME)
    parser = argparse.ArgumentParser(
                    prog='Colbertv2-Search',
                    description='ColBERT IR System')
    
    parser.add_argument('query', type=str, help='Search query')
    args = parser.parse_args()
    query = args.query
    searcher.search(query)
