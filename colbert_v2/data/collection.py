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
    def ensure_proper_header(input_file):
        with open(input_file, 'r') as f_in:
            lines = f_in.readlines()

        # Ensure the first line is exactly 'id\ttext' and the rest remains unchanged
        if not lines[0].strip() == "id\ttext":
            lines.insert(0, "id\ttext\n")
        
        # Write the updated content back to the same file
        with open(input_file, 'w') as f_out:
            f_out.writelines(lines)

    @staticmethod
    def add_missing_headers(input_file):
        with open(input_file, 'r') as f_in:
             lines = f_in.readlines()  # Store the contents of the file

        with open(input_file, 'w') as f_out:
            # Check if the first line has the 'id' and 'text' headers
            if not lines[0].startswith("id\ttext"):
                f_out.write("id\ttext\n")  # Write the header if missing
                lines = lines[1:]  # Skip the first line since we added the header

            # Update the 'id' to match the line number and write the content back
            for line_idx, line in enumerate(lines):
                # Split the line by tab, replace the 'pid' with the line number
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    f_out.write(f"{line_idx}\t{parts[1]}\n")



    def save_new_document(self, new_document):
        with open(self.path, 'a') as f:
            f.write(f"\n{new_document}")
