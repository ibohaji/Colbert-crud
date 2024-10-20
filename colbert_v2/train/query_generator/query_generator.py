import argparse
import json
import os
from collections import defaultdict
from itertools import count
from typing import Dict
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from ...config import MetaData


class QueryGenerator:
    def __init__(self, model_name:str, input_documents, output_path = None, reindex = False)->None:
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_path = output_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.input_documents =  self.index_document(input_documents) if reindex else self._load_documents(input_documents)
        self.counter = count()
    


    def load_jsonl(self, input_file:str)->Dict:
        data = []
        with open(args.input_documents) as f:
            for line in f:
                doc = json.loads(line)
                data.append(doc)
        return data
    
    def load_json(self, input_file:str)->Dict:
        data = []

        with open(input_file) as f:
            json_data = json.load(f)

        for doc in json_data:
            data.append(doc)
        return data
    




    def _load_documents(self, input_documents:str)->Dict:

        if input_documents.endswith('.jsonl'):
            data = self.load_jsonl(input_documents)

        elif input_documents.endswith('.json'):
            data = self.load_json(input_documents)

        else:
            raise ValueError("Input file must be a .json or .jsonl file")
        
        return data

    def index_document(self, documents:Dict)->Dict:
        if  isinstance(documents, str):
            documents = self._load_documents(documents)
        return { idx: doc['text'] for idx, doc in enumerate(documents) }

    def generate_query_ids(self, _range):
        return [next(self.counter) for _ in range(_range)]

    @classmethod
    def _removeNonAscii(s:str) -> str : return "".join(i for i in s if ord(i) < 128)

    def generate_queries(self,batch_size:int = 16)->None:
        generated_queries = {}
        qrel = defaultdict(list)
        doc_items = [(doc_id, doc_text) for doc_id,doc_text in self.documents.values()]

        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
        else:
            self.output_path = "generated_query_data"
            os.makedirs(self.output_path, exist_ok=True)

            for start_idx in tqdm(range(0,len(doc_items),batch_size), desc="Generating queries"):
                batch_docs = doc_items[start_idx:start_idx+batch_size]

                for doc_id, doc in batch_docs:
                    print('\nThe batch docs are\n,',batch_docs)
                    print('Doc ids are \n',doc_id)
                    print('Docs are \n',doc)
                    input_ids = self.tokenizer.encode(
                        doc,
                        max_length=512,
                        truncation=True,
                        return_tensors='pt'
                        ).to(self.device)

                    outputs = self.model.generate(
                        input_ids=input_ids,
                        max_length=64,
                        do_sample=True,
                        top_p=0.95,
                        num_return_sequences=5
                    )

                queries = [self._clean_text(self.tokenizer.decode(output, skip_special_tokens=True)) for output in outputs]
                queries = self.remove_duplicates(queries)
                queries_id = self.generate_query_ids(len(queries))
                for _qid, query_text in zip(queries_id, queries):

                    generated_queries[_qid] = query_text
                    qrel[_qid] = doc_id
                doc_text_clean = self._clean_text(doc)


        self.save_queries_to_json(generated_queries, f"{self.output_path}/generated_queries.json")
        self.save_queries_to_json(qrel, f"{self.output_path}/qrel.json")
        self.generate_split_queries(generated_queries)
        return generated_queries


    def generate_split_queries(self, generated_queries, split_size=2):
        # Hold some of the generated queries for test, remove them from train. keep the indices and id as they were.
        test_queries = {}
        for i in range(0, len(generated_queries), split_size):
            for j in range(i, i+split_size):
                test_queries[j] = generated_queries.pop(j)

        self.save_queries_to_json(generated_queries, os.path.join(self.output_path, 'train_queries.json'))
        self.save_queries_to_json(test_queries, os.path.join(self.output_path, 'test_queries.json'))

    def remove_duplicates(self, generated_queries:list):
        return list(set([query.lower() for query in generated_queries]))

    def save_queries_to_json(self, generated_queries, output_file):
        # Save generated queries to a JSON file
        with open(output_file, "w") as f:
            json.dump(generated_queries, f, indent=2)

    def save_queries_list(self, generated_queries, output_file):
        raise NotImplementedError

    def process_documents(self, document_folder, output_file):
        raise NotImplementedError

    def _clean_text(self, text):
        """Utility function to clean text by removing tabs and non-ASCII characters."""
        return text.replace("\t", " ").encode("ascii", "ignore").decode().strip()

        
if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_documents",
                         type=str,
                         required=True,
                         help="Folder containing the documents"
                           )
    parser.add_argument("--output_file",
                         type=str,
                         default=None,
                         help="Folder to store the documents"
                         )
    parser.add_argument("--model_name",
                         type=str,
                         default='doc2query/S2ORC-t5-base-v1',
                         help="Folder containing the documents"
                         )
    parser.add_argument("--top_p",
                        type=int,
                        default=0.95
                        )
    parser.add_argument("--num_genq",
                        type=int,
                        default=5,
                        help="Number of queries to generate per document"
                        )

    args = parser.parse_args()
    Qgen = QueryGenerator(args.model_name, args.input_documents)
    data = []

    MetaData().update(top_p = args.top_p, num_genq = args.num_genq)
    Qgen.generate_queries(data)
