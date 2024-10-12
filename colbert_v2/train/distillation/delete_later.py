import json 


mapping = {}
new_data = {}

with open('colbert_v2/datasets/scifact/corpus.jsonl') as f:
    for idx, line in enumerate(f): 
        data = json.loads(line)
        mapping[idx] = data['_id']
        new_data[idx] = data 


with open('mapping_collection_ids.json', 'w') as f:
    f.write(json.dumps(mapping, indent=4))

with open('collection_data_altered.json', 'w') as f:
    f.write(json.dumps(new_data, indent=4))
    
