import json
import logging


class GenQueryData:
    def __init__(self, generated_queries_path):
        self.genqueries = generated_queries_path
        self.queries_dict = self.load_data()

    def load_data(self):
        with open(self.genqueries) as f:
            queries = json.load(f)

        queries_dict = {}
        for doc_id, query_list in queries.items():
            for query_text in query_list:
                query_id = f"Q{len(queries_dict) + 1}"
                queries_dict[query_id] = {"text":query_text, "doc_id": doc_id}

        return queries_dict


class CollectionData:
    def __init__(self, collection_path):
        self.collection_path = collection_path
        self.collection_dict = list(self.load_jsonl(collection_path))
        self.collection_pids = self.make_collection_pids()
        logging.info(f"Loaded {len(self.collection_dict)} documents from '{collection_path}'.")

    def load_jsonl(self, path):
        """Loads documents from a JSONL file."""
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    yield {
                        '_id': entry['_id'],
                        'text': entry['text'],
                        'title': entry['title']
                    }
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON line: {e}")
                    continue

    def make_collection_pids(self):
        """Creates a list of all document IDs."""
        return [doc['_id'] for doc in self.collection_dict]

