import argparse
import csv
import json
import logging
import random
import sys
import os
import pathlib
from beir import util

from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from ..config import MetaData
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

csv.field_size_limit(sys.maxsize)

def tsv_reader(input_filepath):
    reader = csv.reader(open(input_filepath, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    for idx, row in enumerate(reader):
        yield idx, row

def map_back(collection, corpus):
    # Mapping from collection_id (sequential index) to corpus_id
    mapping = {}
    doc_id_keys = list(corpus.keys())  # Make sure corpus.keys() is iterable as a list
    for idx, (doc_id_key, row) in enumerate(zip(doc_id_keys, tsv_reader(collection))):
        mapping[str(idx)] = doc_id_key

    return mapping



def main(dataset, split, data_dir, collection, rankings, k_values):
    #### Provide the data_dir where the corpus has been downloaded and unzipped
      
    if data_dir == None:
            print("Downloading the dataset\n"*10)
            
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
            out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
            data_dir = util.download_and_unzip(url, out_dir)
    
    #### Provide the data_dir where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split=split)
    true_map = map_back(collection, corpus)
    inv_map, results = {}, {}
      #### Document mappings (from original string to position in tsv file ####
    for idx, row in tsv_reader(collection):
        inv_map[str(idx)] = row[1]

    #### Results ####
    for _, row in tsv_reader(rankings):
        qid, doc_id, rank = row[0], row[1], int(row[2])
        if qid != true_map[str(doc_id)]:
            if qid not in results:
                results[qid] = {true_map[str(doc_id)]: 1 / (rank + 1)}
            else:
                results[qid][true_map[str(doc_id)]] = 1 / (rank + 1)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric='mrr')

    #### Print top-k documents retrieved ####
    top_k = 10

    query_id, ranking_scores = random.choice(list(results.items()))
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    logging.info("Query : %s\n" % queries[query_id])

    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        # Format: Rank x: ID [Title] Body
        logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))

    colbert_metrics = {
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,   
        "Precision": precision,
    }

    MetaData().update(colbert_metrics = colbert_metrics)
    with open("colbert_metrics.json", "w") as f:
        json.dump(colbert_metrics, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="BEIR Dataset Name, eg. nfcorpus")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--data_dir', type=str, default=None, help='Path to a BEIR repository (incase already downloaded or custom)')
    parser.add_argument('--collection', type=str, help='Path to the ColBERT collection file')
    parser.add_argument('--rankings', required=True, type=str, help='Path to the ColBERT generated rankings file')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1,3,5,10,100,1000])
    args = parser.parse_args()
    main(args.dataset, args.split, args.data_dir, args.collection, args.rankings, args.k_values)