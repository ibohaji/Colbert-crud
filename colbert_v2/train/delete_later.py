import json 





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


with open('qids.json'):
    qids = json.load(f)

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
    