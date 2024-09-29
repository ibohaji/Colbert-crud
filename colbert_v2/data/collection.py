# data/collection.py
class Collection:
    def __init__(self, path):
        self.path = path

    def load_collection(self):
        with open(self.path, 'r') as f:
            documents = [line.strip() for line in f.readlines()]
        return documents

    def save_new_document(self, new_document):
        with open(self.path, 'a') as f:
            f.write(f"\n{new_document}")
