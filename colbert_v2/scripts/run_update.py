# scripts/run_update.py
from config import Config
from models.searcher import ColBERTSearcher
from models.updater import ColBERTUpdater

if __name__ == "__main__":
    searcher = ColBERTSearcher(Config.INDEX_NAME)
    updater = ColBERTUpdater(Config(), searcher)

    # Adding new documents
    new_docs = ["New document 1", "New document 2"]
    updater.add_documents(new_docs)

    # Removing documents by ID
    remove_ids = [0, 1]
    updater.remove_documents(remove_ids)
