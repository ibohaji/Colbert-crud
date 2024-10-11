from colbert.infra import Run
from colbert.infra.config import RunConfig
from .scorer import Scorer
from collections import defaultdict
from ...custom.data_organizer import CollectionData, GenQueryData
import argparse
import json 
import tqdm
import ujson




def main(qid, pid, collection, queries):
    with Run().context(RunConfig(nranks=2)):

        scorer = Scorer(queries=queries, collection=collection)
        distillation_scores = scorer.launch(qids, pids)
        scores_by_qid = defaultdict(list)


    for qid, pid, score in tqdm.tqdm(zip(qids, pids, distillation_scores)):
        scores_by_qid[qid].append((score, pid))

        with Run().open('distillation_scores.json', 'w') as f:
            for qid in tqdm.tqdm(scores_by_qid):
                obj = (qid, scores_by_qid[qid])
                f.write(ujson.dumps(obj) + '\n')

            output_path = f.name
            print(f"Saved distillation scores to {output_path}")


if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--qid_path', type=str, required=True)
    parser.add_argument('--pid_path', type=str, required=True)
    parser.add_argument('--collection_path', type=str, required=True)
    parser.add_argument('--queries_path', type=str, required=True)

    args = parser.parse_args()

    queries = GenQueryData(args.queries_path)
    collection = CollectionData(args.collection_path)
    
    queries = queries.queries_dict
    collection = collection.collection_dict
    collection = {doc['_id']: doc['title'] + doc['text'] for doc in collection}

    with open(args.qid_path, 'r') as f:
        qids = [line.strip() for line in f.readlines()]

    # For pids
    with open(args.pid_path, 'r') as f:
        pids = [line.strip() for line in f.readlines()]

    main(qids, pids, collection, queries)


