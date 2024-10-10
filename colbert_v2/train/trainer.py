# colbert_v2/train/trainer.py
from ...ColBERT.colbert.infra.run import Run
from ...ColBERT.colbert.infra.config import ColBERTConfig, RunConfig
from ...ColBERT.colbert import Trainer
from ...ColBERT.colbert import Trainer
import argparse



def train(self, checkpoint, nrank, triples, queries, collection):
    with Run().context(RunConfig(nranks=nrank)):
            triples = triples # `wget https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/resolve/main/examples.json?download=true` (26GB)
            queries = queries # TSV file with query_id, query_text
            collection = collection # TSV file with doc_id, doc_text

            config = ColBERTConfig(bsize=32, lr=1e-05, warmup=20_000, doc_maxlen=180, dim=128, attend_to_mask_tokens=False, nway=64, accumsteps=1, similarity='cosine', use_ib_negatives=True)
            trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)
            trainer.train(checkpoint='colbert-ir/colbertv1.9')
            # save the model 


if __name__=="__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--nrank', type=int, required=True)
        parser.add_argument('--triples', type=str, required=True)
        parser.add_argument('--queries', type=str, required=True)
        parser.add_argument('--collection', type=str, required=True)


            