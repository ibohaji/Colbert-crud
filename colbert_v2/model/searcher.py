# models/searcher.py
from colbert import Searcher
from colbert.infra import Run, RunConfig
from config import Config

class ColBERTSearcher:
    def __init__(self, index_name):
        self.config = Config()

    def search(self, query):
        with Run().context(RunConfig(experiment='searching')):
            searcher = Searcher(index=self.config.INDEX_NAME, config=self.config)
            results = searcher.search(query, k=20)  # Retrieve top 20 results

        for passage_id, passage_rank, passage_score in zip(*results):
            print(f"[{passage_rank}] {passage_score:.1f} - {searcher.collection[passage_id]}")
