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
            first_line = f_in.readline().strip()
            headers = first_line.split('\t')
            
            if headers[0] != "id" or len(headers) < 2 or headers[1] != "text":
                with open(input_file, 'w') as f_out:
                    f_out.write("id\ttext\n")
                    
                    if headers[0] != "id":
                        f_out.write(first_line + '\n')
                    
                    for line in f_in:
                        f_out.write(line)
            else:
                with open(input_file, 'w') as f_out:
                    f_out.write(first_line + '\n')
                    for line in f_in:
                        f_out.write(line)


    def save_new_document(self, new_document):
        with open(self.path, 'a') as f:
            f.write(f"\n{new_document}")
