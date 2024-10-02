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
import argparse 
import os 

class ColBERTSearcher:
    def __init__(self, index_name, queries_path):
        self.config = Config()
        self.queries_path = queries_path

    def search(self, ranking_output):
        with Run().context(RunConfig(nranks=1, experiment='experiments')):
            config = ColBERTConfig(root="experiments")
            searcher = Searcher(index=self.config.INDEX_NAME, config=config)
            queries = Queries(queries)
            ranking = searcher.search_all(queries, k=100)  
            output_path = os.makdirs(ranking_output, exist_ok=True)
            ranking.save('scifact.nbit=2.ranking.tsv')

if __name__ == "__main__":
    ## parse args and call searcher
    parser = argparse.ArgumentParser()
    parser.add_argument('--Queries', type=str, default="crud.colbert.index")
    parser.add_argument('--output_path', type=str, help='Path to store rankings')
    args = parser.parse_args()
    queries = args.Queries 
    output = args.output_path
    config = Config()
    searcher = ColBERTSearcher(config.INDEX_NAME, queries)
    searcher.search(output)
