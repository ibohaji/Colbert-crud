import os
import json
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Dict,List 
from pathlib import Path 
import torch
import argparse


class QueryGenerator:
    def __init__(self, model_name:str, output_path = None)->None:
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_path = output_path
        self.model.to(self.device)



    @classmethod 
    def _removeNonAscii(s:str) -> str : return "".join(i for i in s if ord(i) < 128)

    def generate_queries(self, documents:Dict,batch_size:int = 16)->None:
        generated_queries = {}
        doc_items = ["Title:" + doc['title'] + "\t" + doc['text'] for doc in documents]


        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True) 
        else: 
            self.output_path = "generated_query_data"
            os.makedirs(self.output_path, exist_ok=True)

        with open(f"{self.output_path}/generated_documents.tsv","w") as f: 
            for start_idx in tqdm(range(0,len(doc_items),batch_size), desc="Generating queries"):
                batch_docs = doc_items[start_idx:start_idx+batch_size]

                for doc_id, doc in batch_docs:
                    input_ids = self.tokenizer.encode(
                        doc['text'], 
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
                generated_queries[doc_id] = queries 
                doc_text_clean = self._clean_text(doc['text'])

                for query in queries:
                    f.write(f"{query}\t{doc_text_clean}\n")
        

        self.save_queries_to_json(generated_queries, f"{self.output_path}/generated_queries.json")
        return generated_queries


    def save_queries_to_json(self, generated_queries, output_file):
        # Save generated queries to a JSON file
        with open(output_file, "w") as f:
            json.dump(generated_queries, f, indent=2)

    def save_queries_list(self, generated_queries, output_file):
        raise NotImplementedError

    def process_documents(self, document_folder, output_file):
        # Main method to process documents and generate queries
        raise NotImplementedError

    def _clean_text(self, text):
        """Utility function to clean text by removing tabs and non-ASCII characters."""
        return text.replace("\t", " ").encode("ascii", "ignore").decode().strip()
    


if __name__ =="__main__": 
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--input_documents", type=str, required=True, help="Folder containing the documents") 
    parser.add_argument("--output_file", type=str, default=None, help="Folder to store the documents") 
    parser.add_argument("--model_name", type=str, default='doc2query/S2ORC-t5-base-v1', help="Folder containing the documents") 
    
    args = parser.parse_args() 
    Qgen = QueryGenerator(args.model_name)
    data = {}
    with open(args.input_documents) as f:
        #jsonl file containing the documents
        for line in f:
            doc = json.loads(line)
            data.append(doc)


    Qgen.generate_queries(data) 
