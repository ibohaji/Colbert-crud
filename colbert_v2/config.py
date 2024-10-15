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


def Singleton(cls):
    _instances = {}
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls)
        return _instances[cls]


class MetaData(metaclass=Singleton):
    def __init__(self):
        self.EXPERIMENT_ID = None
        self.NWAY = None
        self.BSIZE = None
        self.NDCG = None
        self.SAMPLING_METHOD = ''
        self.LR = None
        self.WARMUP = None
        self.BASE_MODEL = ''
        self.EPOCHS = None
        self.top_p =None
        self.num_genq = None
        self.NUM_QUERIES = None
        self.NUM_COLLECTION = None
        self.GPU_TYPE = ''
        self.NUM_GPUS = None
        self.CHECKPOINT_PATH = ''
        self.TRAINING_TIME = None
        self.DISTILLATION_TIME = None
        self.EVALUATION_TIME =None
        self.SEED = None



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



