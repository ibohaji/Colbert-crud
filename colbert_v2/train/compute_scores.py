# colbert_v2/train/compute_scores.py

""" 
This script is used to prepare the data for distillation. 
It computes the scores for the documents in the collection and the queries in the dataset using a cross encoder,
for which the scores will be distilled into the student model according to colbert_v2.
"""

from colbert.distillation.scorer import Scorer
from colbert.data.dataset import Queries, Collection 
from colbert_v2.train.sampler import HardNegativesSampler, SoftNegativeSampler
from ..custom.data_loader import GenQueryData
import argparse 

def compute_rank(corpus_path, generated_queries_path):
    
    queries_dict = GenQueryData(generated_queries_path)

    neg_sampler = HardNegativesSampler()


if "__name__" == "__main__":
    pass