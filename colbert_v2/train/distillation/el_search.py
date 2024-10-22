import logging
import subprocess
import os
import time
import requests
from elasticsearch import Elasticsearch, helpers
from contextlib import ContextDecorator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EsSearcher(ContextDecorator):
    def __enter__(self):
        """This is called when entering the context (when Elasticsearch starts)."""
        try:
            # Start Elasticsearch in the background and capture the process
            self.es_process = self.start_elasticsearch()
            host = self.get_host_ip()

            # Wait until Elasticsearch is ready
            self.wait_for_elasticsearch(host)

            # Connect to Elasticsearch
            self.es = Elasticsearch(f"http://{host}:9200", timeout=30)
            self.index_name = 'documents'
            logger.info(f"Connected to Elasticsearch at {host}")

            # Delete the index if it exists
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing index '{self.index_name}'")

            return self  # Allow chaining of methods
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            raise e


    def __exit__(self, exc_type, exc_val, exc_tb):
        """This is called when leaving the context (cleanup/termination)."""
        self.terminate_es()
        if exc_type:
            logger.error(f"Exception occurred: {exc_type}, {exc_val}")
            return False  # Propagate the exception
        return True

    def retrieve(self, index, query_text, positive_doc_id, num_results=10):
         search_body = {
            "size": num_results,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["title", "text"]                        }
                    },
                    "must_not": {
                        "ids": {
                            "values": [positive_doc_id]
                        }
                    }
                }
            }
        }

         response = self.es.search(index=self.index_name, body=search_body)
         return response['hits']['hits']




    def index_documents(self, collection_data):
        """Indexes documents into Elasticsearch using the bulk helper."""
        # Define the mapping for the index
        mapping = {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "text": {"type": "text"}
                }
            }
        }

        try:
            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created index '{self.index_name}' with mapping.")
        except Exception as e:
            logger.error(f"Error creating index '{self.index_name}': {e}")
            raise e

        # Prepare actions for bulk indexing
        actions = [
            {
                "_index": self.index_name,
                "_id": doc['_id'],
                "_source": {
                    "title": doc['title'],
                    "text": doc['text']
                }
            }
            for doc in collection_data
        ]

        try:
            success, failed = helpers.bulk(
                self.es,
                actions,
                refresh=True,
                raise_on_error=False,
                stats_only=False
            )
            logger.info(f"Successfully indexed {success} documents.")
            if failed:
                logger.warning(f"Failed to index {len(failed)} documents.")
                for failure in failed:
                    logger.warning(f"Failed document: {failure}")
        except Exception as e:
            logger.error(f"Error during bulk indexing: {e}")
            raise e
        


    def start_elasticsearch(self):
        print("Starting Elasticsearch...")
        es_process = subprocess.Popen(
            ['/zhome/3a/7/145702/elasticsearch-8.5.0/bin/elasticsearch'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return es_process

    def wait_for_elasticsearch(self, host, max_retries=5):
        retries = 0
        logger.info(f"Waiting for Elasticsearch to be available at {host}...")

        while retries < max_retries:
            try:
                response = requests.get(f"http://{host}:9200")
                if response.status_code == 200:
                    logger.info("Elasticsearch is ready!")
                    return
            except requests.exceptions.ConnectionError:
                retries += 1
                logger.warning(f"Elasticsearch not ready, retrying... ({retries}/{max_retries})")
                time.sleep(5)

        raise ConnectionError(f"Failed to connect to Elasticsearch after {max_retries} attempts.")

        raise ConnectionError(f"Failed to connect to Elasticsearch after {max_retries} attempts.")
    

    def get_host_ip(self):
        host_ip = subprocess.getoutput("hostname -I").split()[0]
        logger.info(f"Detected host IP: {host_ip}")
        return host_ip


    
    def terminate_es(self):
        if hasattr(self, 'es_process') and self.es_process:
            logger.info("Terminating Elasticsearch...")
            self.es_process.terminate()
            self.es_process.wait()
            logger.info("Elasticsearch terminated.")



