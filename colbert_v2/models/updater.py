# models/updater.py
from colbert import IndexUpdater

class ColBERTUpdater:
    def __init__(self, config, searcher):
        self.config = config
        self.searcher = searcher

    def add_documents(self, new_docs):
        index_updater = IndexUpdater(config=self.config, searcher=self.searcher, checkpoint=self.config.CHECKPOINT)
        index_updater.add(new_docs)
        index_updater.persist_to_disk()
        print(f"Documents added to index: {new_docs}")

    def remove_documents(self, ids_to_remove):
        index_updater = IndexUpdater(config=self.config, searcher=self.searcher, checkpoint=self.config.CHECKPOINT)
        index_updater.remove(ids_to_remove)
        index_updater.persist_to_disk()
        print(f"Documents removed from index: {ids_to_remove}")
