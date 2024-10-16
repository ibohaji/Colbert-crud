# models/searcher.py
import argparse
import json
import os
from time import time
from ..custom.execution_monitor import monitor_gpu
from colbert import Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig

from ..config import Config, MetaData


class ColBERTSearcher:
    def __init__(self, index_name, queries_path):
        self.config = Config()
        self.queries_path = queries_path

    @monitor_gpu
    def search(self, ranking_output):
        start_time = time()
        with Run().context(RunConfig(nranks=1, experiment='experiments')):
            start_time = time()
            config = ColBERTConfig(root="experiments")
            searcher = Searcher(index=self.config.INDEX_NAME, config=config, checkpoint = self.config.CHECKPOINT)
            queries = Queries(self.queries_path)
            ranking = searcher.search_all(queries, k=1000)
            os.makedirs(ranking_output, exist_ok=True)
            out_file = os.path.join(ranking_output, "scifact_fine_tuned_ranking.tsv")
            ranking.save(out_file)

        total_time = time() - start_time
        print(f"Search completed successfully in {total_time} seconds")
        MetaData().update(Search_time=total_time)

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
