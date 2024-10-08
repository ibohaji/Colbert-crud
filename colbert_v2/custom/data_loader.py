
import json 

class GenQueryData: 

    def __init__(self, generated_queries_path, corpus_path): 
        self.genqueries = generated_queries_path
        self.corpus_path = corpus_path
        self.queries_dic = self.load_data() 

    def load_data(self): 
        # json file
        queries = {} 
        document_id = {} 
        with open(self.genqueries, 'r') as f:
            queries = json.load(f)

        queries_dict = {query: doc_id for doc_id, query_list in queries.items() for query in query_list}
        return queries_dict


class CollectionData: 
    def __init__(self, collection_path): 
        self.collection_path = collection_path 
    

    def load_jsonl(self, path):
        raise NotImplementedError