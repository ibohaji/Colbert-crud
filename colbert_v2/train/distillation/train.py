from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer
import os
import json
import logging
import argparse
from collections import defaultdict
import tqdm
import ujson
import logging
from logging import getLogger
import csv




def json_to_tsv(input_file, output_file):
    is_jsonl = input_file.endswith('.jsonl')

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        writer.writerow(['id', 'text'])  

        if is_jsonl:
            for line in infile:
                data = json.loads(line)
                writer.writerow([data.get('id'), data.get('text')])
        else:
            data = json.load(infile)
            for item in data:
                writer.writerow([item.get('id'), item.get('text')])



def run_distillation(triples, queries, collection):
    

    with Run().context(RunConfig(nranks=1)):

        triples = triples
        queries = queries  # '/path/to/MSMARCO/queries.train.tsv'
        collection = collection       #'/path/to/MSMARCO/collection.tsv'

        config = ColBERTConfig(bsize=32, lr=1e-05, warmup=20_000, doc_maxlen=180, dim=128, attend_to_mask_tokens=False, nway=64, accumsteps=1, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)
        trainer.train(checkpoint='colbert-ir/colbertv1.9')
        




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--triples_path', type=str, required=True)
    parser.add_argument('--queries', type=str, required=True)
    parser.add_argument('--collection', type=str, required=True)

    args = parser.parse_args()
    triples = args.triples_path
    queries = args.queries
    collection = args.collection

    output_path_queries = 'queries.tsv'
    output_path_collection = 'collection.tsv'

    json_to_tsv(queries, output_path_queries)
    json_to_tsv(collection, output_path_collection)

    run_distillation(triples, output_path_queries, output_path_collection)

