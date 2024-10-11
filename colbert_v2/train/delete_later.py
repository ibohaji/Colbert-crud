import json 



"""

with open('queries_generated_mapping.json') as f:
    queries = json.load(f)


altered_query = {}

for query_id, query_text in queries.items():
    print(f"Query ID: {query_id}")
    print(f"Query Text: {query_text}")
    if query_id.startswith('Q'):
        new_id = query_id.split('Q')[1]
    else:
        raise ValueError("Invalid query ID")
    
    altered_query[new_id] = query_text
    

with open('queries_generated_mapping_altered.json', 'w') as f:
    f.write(json.dumps(altered_query, indent=4))


with open('qids.json', 'r') as f:
    qids = [line.strip().strip('"') for line in f.readlines()]


new_qids = []

for qid in qids:
    if qid.startswith('Q'):
        new_id = qid.split('Q')[1]
    else:
        raise ValueError("Invalid query ID")
    new_qids.append(new_id)


with open('qids_altered.json', 'w') as f:
    f.write(json.dumps(new_qids, indent=4))


"""
input_file = 'distillation_scores.json'
output_file = 'altered_distillation.json'

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for line_number, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        try:
            data = json.loads(line)
            key = data[0]
            if not key.startswith('Q'):
                print(f"Warning: Line {line_number} key does not start with 'Q': {key}")
                new_key = key  # or handle as needed
            else:
                new_key = key[1:]  # Remove the 'Q'
            new_entry = [new_key, data[1]]
            json.dump(new_entry, fout)
            fout.write('\n')  # Write each JSON array on a new line
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_number}: {e}")
            # Handle or skip the malformed line as needed
