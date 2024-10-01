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

            # Check if the first line has the 'id' and 'text' headers
            if not first_line.startswith("id\ttext"):
                with open(input_file, 'w') as f_out:
                    f_out.write("id\ttext\n")  # Add header
                    f_out.write(first_line + '\n')  # Write the first line as data
                    f_out.writelines(f_in)  # Write the rest of the file
            else:
                # If the header exists, just copy the file as-is
                with open(input_file, 'w') as f_out:
                    f_out.write(first_line + '\n')
                    f_out.writelines(f_in)



    def save_new_document(self, new_document):
        with open(self.path, 'a') as f:
            f.write(f"\n{new_document}")
