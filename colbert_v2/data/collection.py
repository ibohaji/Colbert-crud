# data/collection.py
class FileProcessor:
    def __init__(self, path):
        self.path = path

    def load_collection(self):
        documents = []
        with open(self.path, 'r') as f:
            for line in f:
                columns = line.strip().split('\t')
                documents.append(columns)
        return documents
    
    @staticmethod
    def add_missing_headers(input_file):
        with open(input_file, 'r') as f_in:
            lines = f_in.readlines()  # Store the contents of the file

        # Open the file again in write mode
        with open(input_file, 'w') as f_out:
            # Check if the first line has the 'id' and 'text' headers
            if not lines[0].startswith("id\ttext"):
                f_out.write("id\ttext\n")  # Write header if missing
            
            # Write back the original content
            f_out.writelines(lines)




    def save_new_document(self, new_document):
        with open(self.path, 'a') as f:
            f.write(f"\n{new_document}")
