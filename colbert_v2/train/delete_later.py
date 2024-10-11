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
["Q19",[[8.6171875,"343052"],[-4.3125,"14834714"],[-2.154296875,"24998637"],[2.66796875,"7948486"]]]
["Q20",[[2.0078125,"343052"],[1.849609375,"1360607"],[-6.19140625,"24998637"],[-9.7265625,"14834714"]]]
["Q21",[[4.703125,"427082"],[-5.609375,"4932668"],[-7.2265625,"6148876"],[3.033203125,"38727075"]]]
""""""
with open('distillation_scores.json', 'r') as f:
    input_data = json.load(f)

altered_data = {}

for key, values in input_data.items():
    new_key = int(key.split('Q')[1])  
    altered_data[new_key] = values

with open('altered_distillation.json', 'w') as f:
    json.dump(altered_data, f, indent=4)
