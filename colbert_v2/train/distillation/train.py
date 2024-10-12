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

def convert_json_file_to_jsonl(input_json_path, output_jsonl_path):
    with open(input_json_path, 'r') as json_file:
        input_json = json.load(json_file)
    
    with open(output_jsonl_path, 'w') as jsonl_file:
        for key, value in input_json.items():
            jsonl_file.write(json.dumps({key: value}) + '\n')


def json_to_tsv(input_file, output_file):
    is_jsonl = input_file.endswith('.json')

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter='\t')

        if is_jsonl:
            for line in infile:
                data = json.loads(line)
                if isinstance(data, dict):
                    writer.writerow([data.get('id'), data.get('text')])
        else:
            data = json.load(infile)
            if isinstance(data, list):
                for item in data:
                    writer.writerow([item.get('id'), item.get('text')])
            elif isinstance(data, dict):
                for key, value in data.items():
                    writer.writerow([key, value])



def run_distillation(triples_, queries_, collection_):
    

    with Run().context(RunConfig(nranks=1)):
        
        config = ColBERTConfig(bsize=32, lr=1e-05, warmup=20_000, doc_maxlen=180, dim=128, attend_to_mask_tokens=False, nway=64, accumsteps=1, similarity='cosine', use_ib_negatives=True)
        trainer = Trainer(triples=triples_, queries=queries_, collection=collection_, config=config)
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

    output_path_collection = 'collection.tsv'
    output_path_queries = 'queries.json'


    convert_json_file_to_jsonl(queries, output_path_queries)
    run_distillation(triples, output_path_queries, collection)

