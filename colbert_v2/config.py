# config.py
from colbert.infra import ColBERTConfig

class Config:
    CHECKPOINT = "colbert-ir/colbertv2.0"  # Pretrained ColBERT V2 model
    DOC_MAXLEN = 300  # Maximum length of each document
    NBITS = 2  # Embedding cell bits (for IVF indexing)
    KMEANS_NITERS = 8  # KMeans iterations for FAISS clustering
    NCELLS = 4000  # Number of cells for coarse search
    NDOCS = 1000  # Number of documents to retrieve
    INDEX_NAME = "crud.colbert.index"
    COLLECTION_PATH = "colbert_v2/data/collection.tsv"
    QUERIES_PATH = "colbert_v2/data/queries.tsv"