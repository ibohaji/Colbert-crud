# models/searcher.py
import argparse
import json
import os
from time import time

from colbert import Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig

from ..config import Config, MetaData


class ColBERTSearcher:
    def __init__(self, index_name, queries_path):
        self.config = Config()
        self.queries_path = queries_path

    def search(self, ranking_output):
        start_time = time()
        with Run().context(RunConfig(nranks=1, experiment='experiments')):
            start_time = time()
            config = ColBERTConfig(root="experiments")
            searcher = Searcher(index=self.config.INDEX_NAME, config=config)
            queries = Queries(self.queries_path)
            ranking = searcher.search_all(queries, k=1000)
            output_path = os.makedirs(ranking_output, exist_ok=True)
            ranking.save('scifact.nbit=2.ranking.tsv')

        total_time = time() - start_time

        MetaData.update(Search_time=total_time)

        with open("search_time.json", "w") as f:
            json.dump({"search_time": total_time}, f, indent=2)

if __name__ == "__main__":
    ## parse args and call searcher
    parser = argparse.ArgumentParser()
    parser.add_argument('--Queries', type=str, default="crud.colbert.index")
    parser.add_argument('--output_path', type=str, help='Path to store rankings')
    parser.add_argument('--experiment', type=str, default="scifact")
    parser.add_argument('--index_name', type=str, default="experiments")
    args = parser.parse_args()
    queries = args.Queries
    output = args.output_path
    config = Config()
    searcher = ColBERTSearcher(args.index_name, queries)
    searcher.search(output)
