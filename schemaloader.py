import json

from path import schema_path

predicate2id, id2predicate = {}, {}

with open(schema_path, encoding = 'utf-8') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)
