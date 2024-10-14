# config.py
#from colbert.infra import ColBERTConfig
from dataclasses import dataclass
import json 

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



@dataclass
class MetaData():
    """ Class to store all the parameters of the experiments """
    NWAY: int
    BSIZE: int
    NDCG: float 
    SAMPLING_METHOD: str 
    LR: float
    WARMUP: int
    BASE_MODEL: str
    EPOCHS: int 
    NUM_QUERIES: int
    NUM_COLLECTION: int
    GPU_TYPE: str
    NUM_GPUS: int
    CHECKPOINT_PATH: str
    TRAINING_TIME: float
    DISTILLATION_TIME: float
    EVALUATION_TIME: float
    SEED: int


def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class MetaData:
    def __init__(self):
        self.EXPERIMENT_ID = None
        self.NWAY = 0
        self.BSIZE = 0
        self.NDCG = 0.0
        self.SAMPLING_METHOD = ''

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    
@singleton
class MetaData:
    def __init__(self):
        self.EXPERIMENT_ID = None
        self.NWAY = 0
        self.BSIZE = 0
        self.NDCG = 0.0
        self.SAMPLING_METHOD = ''

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


    def save_results(self, file_path='/experiment_results/'):
        """ Save the metadata to a JSON file """
        data = self.__dict__
        file_name = file_path + self.EXPERIMENT_ID +'_'+ 'metadata.json'
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"MetaData saved to {file_path}")



