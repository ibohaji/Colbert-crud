# config.py
from colbert.infra import ColBERTConfig
from dataclasses import dataclass

@dataclass
class Config:
    CHECKPOINT = "colbert-ir/colbertv2.0"  
    DOC_MAXLEN = 300  
    NBITS = 2  
    KMEANS_NITERS = 8  # KMeans iterations for FAISS clustering
    NCELLS = 4000  # Number of cells for coarse search
    NDOCS = 1000  # Number of documents to retrieve
    INDEX_NAME = "crud.colbert.index"
    COLLECTION_PATH = "colbert-datasets/scifact/collection.tsv"
    QUERIES_PATH = "colbert_v2/data/queries.tsv"