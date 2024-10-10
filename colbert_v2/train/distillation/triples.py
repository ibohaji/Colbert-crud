from colbert.data import Ranking
from ....custom.data_organizer import GenQueryData, CollectionData
from sampler import HardNegativesSampler
import json
import ujson
import logging
import argparse


def save_triples(scored_triples, output_path):
    with open(output_path, 'w') as f:
        for triple in scored_triples:
            ujson.dump(triple, f)
            f.write('\n')
    print(f"Saved triples to {output_path}")


def integrate_scores(triples, scores):
    return [(query_id, pos_id, neg_id, score) for (query_id, pos_id, neg_id), score in zip(triples, scores)]


def compute_scores(triples, queries_data, collection_data):
    queries = {qid: query['text'] for qid, query in queries_data.queries_dict.items()}
    collection = {doc['_id']: doc['text'] for doc in collection_data.collection_dict}

    #scorer = Scorer(queries, collection, model='cross-encoder/ms-marco-MiniLM-L-6-v2')

    qids = [triple[0] for triple in triples]
    pids = [triple[1] for triple in triples]
    
  #  scores = scorer.launch(qids, pids)
   # return scores


def generate_triples_with_colbert_logic(queries_data, hard_negatives):
    triples = []
    for query_id, query_data in queries_data.queries_dict.items():
        positive_doc_id = query_data['doc_id']
        negatives = hard_negatives.get(query_id, [])
        for negative_doc_id in negatives:
            triples.append((query_id, positive_doc_id, negative_doc_id))
    return triples


def main(generated_queries_path, collection_path):

    queries_data = GenQueryData(generated_queries_path)
    collection_data = CollectionData(collection_path)

    hard_negatives = HardNegativesSampler(queries=queries_data, collection=collection_data).get_hard_negatives_all(num_negatives=3)

    triples = generate_triples_with_colbert_logic(queries_data, hard_negatives)
    save_triples(triples, 'scored_triples.json')
    print(f"saved triples to 'scored_triples.json")
   # scores = compute_scores(triples, queries_data, collection_data)

   # scored_triples = integrate_scores(triples, scores)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_queries_path", type=str, required=True, help="Path to the generated queries")
    parser.add_argument("--collection_path", type=str, required=True, help="Path to the collection")
    args = parser.parse_args()
    print('starting..')
    main(args.generated_queries_path, args.collection_path)
