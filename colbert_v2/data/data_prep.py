import argparse
import csv
import logging
import os
import pathlib
import random

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm

random.seed(42)
import random


def preprocess(text):
    return text.replace("\r", " ").replace("\t", " ").replace("\n", " ")

def main(dataset, split, data_dir, collection, queries):

    if data_dir == None:
        #### Download .zip dataset and unzip the dataset
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_dir = util.download_and_unzip(url, out_dir)
        logging.info(f"Downloaded {dataset} BEIR dataset: {out_dir}")

    logging.info(f"Coverting {split.upper()} split of {dataset} dataset...")

    #### Provide the data_dir where nfcorpus has been downloaded and unzipped
    corpus, _queries, qrels = GenericDataLoader(data_folder=data_dir).load(split=split)
    corpus_ids = list(corpus)

    #### Create output directories for collection and queries
    os.makedirs("/".join(collection.split("/")[:-1]), exist_ok=True)
    os.makedirs("/".join(queries.split("/")[:-1]), exist_ok=True)

    logging.info(f"Preprocessing Corpus and Saving to {collection} ...")
    with open(collection, 'w') as fIn:
        writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for idx, doc_id in enumerate(tqdm(corpus_ids, total=len(corpus_ids))):
            doc = corpus[doc_id]
            document_text = (preprocess(doc.get("title", "")) + " " + preprocess(doc.get("text", ""))).strip()
            writer.writerow([idx, document_text])  # Include idx (internal PID), doc_id, and text

    logging.info(f"Preprocessing Queries and Saving to {queries} ...")
    with open(queries, 'w') as fIn:
        writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for qid, query in tqdm(_queries.items(), total=len(_queries)):
            writer.writerow([qid, query])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="BEIR Dataset Name, eg. nfcorpus")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--data_dir', type=str, default=None, help='Path to a BEIR repository (incase already downloaded or custom)')
    parser.add_argument('--collection', type=str, help='Path to store BEIR collection tsv file')
    parser.add_argument('--queries', type=str, help='Path to store BEIR queries tsv file')
    args = parser.parse_args()
    main(**vars(args))
