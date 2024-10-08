from rank_bm25 import BM25Okapi

# Prepare BM25 corpus, get hard negatives

class HardNegativesSampler:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def get_scores(self, query):
        return self.bm25.get_scores(query)
    
    def get_hard_negative_pids(self, query_text, positive_pids, collection_pids, num_negatives=3):
        scores = self.get_scores(query_text.split())
        ranked_pids = [pid for pid, _ in sorted(enumerate(scores), key=lambda x: -x[1])]
        hard_negatives = [collection_pids[i] for i in ranked_pids if collection_pids[i] not in positive_pids]
        return hard_negatives[:num_negatives]


class SoftNegativeSampler: 
    def __init__(self):
        raise NotImplementedError