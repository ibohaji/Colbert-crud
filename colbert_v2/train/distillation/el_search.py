from elasticsearch import Elasticsearch, helpers
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EsSearcher:
    def __init__(self, host):
        try:
            self.es = Elasticsearch(host, timeout=30)
            self.index_name = 'documents'
            logger.info(f"Connected to Elasticsearch at {host}")

            # Delete the index if it exists
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing index '{self.index_name}'")
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            raise e
        
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
        
    



