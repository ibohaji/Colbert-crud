# colbert_v2/train/trainer.py
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer

class Trainer:
    def __init__(self, experiment): 
        pass 

    
    def train(self, checkpoint, nrank):
        with Run().context(RunConfig(nranks=nrank)):
                triples = '/path/to/examples.64.json'  # `wget https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/resolve/main/examples.json?download=true` (26GB)
                queries = '/path/to/MSMARCO/queries.train.tsv'
                collection = '/path/to/MSMARCO/collection.tsv'

                config = ColBERTConfig(bsize=32, lr=1e-05, warmup=20_000, doc_maxlen=180, dim=128, attend_to_mask_tokens=False, nway=64, accumsteps=1, similarity='cosine', use_ib_negatives=True)
                trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

                trainer.train(checkpoint='colbert-ir/colbertv1.9')