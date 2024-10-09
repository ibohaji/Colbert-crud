import json
import logging
from ..custom.data_organizer import CollectionData, GenQueryData
from .el_search import EsSearcher
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt')
class HardNegativesSampler:
    def __init__(self, queries, collection):
        self.collection = collection
        self.es = self.setup_es_index()
        self.index_name = 'documents'
        self.queries = queries

    def setup_es_index(self):
        es_searcher = EsSearcher()
        es_searcher.index_documents(self.collection.collection_dict)
        return es_searcher

    def get_hard_negatives_all(self, num_negatives=3):
        hard_negatives = {}
        for query_id, item in self.queries.queries_dict.items():
            query_text = item['text']
            positive_doc_id = item['doc_id']
            negatives = self.get_hard_negative_pids(query_text, positive_doc_id, num_negatives)
            hard_negatives[query_id] = negatives
        return hard_negatives

    def get_hard_negative_pids(self, query_text, positive_doc_id, num_negatives):
        try:
            hits = self.es.retrieve(index=self.index_name, query_text=query_text, positive_doc_id=positive_doc_id, num_results=num_negatives)
            hit_ids = [hit['_id'] for hit in hits]
            logger.info(f"Search Response for Query '{query_text}': {hit_ids}")
            return hit_ids
        except Exception as e:
            logger.error(f"Error searching for hard negatives: {e}")
            return []

       

def main():
    # Sample data for generated queries and documents
    genqueries_data = {
        "doc1": ["What do cats typically eat?", "What kind of food does a cat prefer?"],
        "doc2": ["What animals eat small rodents?", "Which pets hunt mice?"],
        "doc3": ["How do airplanes fly?", "What is the science behind flight?"],
        "doc4": ["What is the capital of France?", "Which city is the capital of France?"],
        "doc5": ["How does photosynthesis work?", "What is the process of photosynthesis?"],
        "doc6": ["What are the health benefits of yoga?", "How does yoga improve health?"],
        "doc7": ["How to cook a perfect steak?", "What is the best way to cook a steak?"]
    }

    # Save the generated queries to a file
    genqueries_path = 'generated_queries.json'
    with open(genqueries_path, 'w') as f:
        json.dump(genqueries_data, f)
        logger.info(f"Saved generated queries to '{genqueries_path}'.")

    # Sample collection data
    collection_data = [
        {"_id": "doc1", "title": "Cats and Their Diet", "text": "Cats generally eat small animals, such as mice, or specially prepared cat food."},
        {"_id": "doc2", "title": "Animals That Eat Rodents", "text": "Various animals like cats and owls hunt small rodents like mice and rats."},
        {"_id": "doc3", "title": "Principles of Flight", "text": "Airplanes fly by generating lift, which is created by the flow of air over the wings."},
        {"_id": "doc4", "title": "Paris, the Capital of France", "text": "Paris is the capital and largest city of France, known for its culture and history."},
        {"_id": "doc5", "title": "Understanding Photosynthesis", "text": "Photosynthesis is the process by which plants convert sunlight into chemical energy."},
        {"_id": "doc6", "title": "Benefits of Yoga", "text": "Yoga offers numerous health benefits, including improved flexibility, strength, and stress reduction."},
        {"_id": "doc7", "title": "Cooking the Perfect Steak", "text": "The perfect steak is cooked by searing it at high temperature and then letting it rest to retain juices."}
    ]

    # Save the collection data to a JSONL file
    collection_path = 'collection.jsonl'
    with open(collection_path, 'w') as f:
        for doc in collection_data:
            f.write(json.dumps(doc) + '\n')
    logger.info(f"Saved collection data to '{collection_path}'.")

    # Load the collection data using CollectionData class
    collection = CollectionData(collection_path)

    # Load the queries using GenQueryData class
    queries = GenQueryData(generated_queries_path=genqueries_path)

    # Initialize HardNegativesSampler with the Elasticsearch client
    sampler = HardNegativesSampler(index_name="documents", queries=queries, collection=collection)

    # Retrieve hard negatives
    hard_negatives = sampler.get_hard_negatives_all()
    print("\nHard Negatives for all queries:")
    print(json.dumps(hard_negatives, indent=2))

if __name__ == "__main__":
    main()
