from ....ColBERT.colbert.distillation.scorer import Scorer
from ....ColBERT.colbert.distillation.ranking_scorer import RankingScorer
from ....ColBERT.colbert.infra import Run
from collections import defaultdict
import wandb
import argparse
import json 
import tqdm
import ujson


@wandb.config()
def main(qid, pid, collection, queries):
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
    parser.add_argument('-- qid_path', type=str, required=True)
    parser.add_argument('-- pid_path', type=str, required=True)
    parser.add_argument('--collection_path', type=str, required=True)
    parser.add_argument('--queries_path', type=str, required=True)

    arg = parser.parse_args()


    with open(arg.collection_path, 'r') as f:
        collection = json.load(f)
    
    with open(arg.queries_path, 'r') as f:
        queries = json.load(f)


    with open(arg.pid_path, 'r') as f:
        pids = json.load(f)
    
    with open(arg.qid_path, 'r') as f:
        qids = json.load(f)


