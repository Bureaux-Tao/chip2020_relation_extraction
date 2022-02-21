import json

import path

with open(path.train_file_path, 'r', encoding = 'utf-8') as f1:
    with open(path.train_file_path, 'r', encoding = 'utf-8') as f2:
        with open(path.train_file_path, 'r', encoding = 'utf-8') as f3:
            max = 0
            for l in f1:
                l = json.loads(l)
                if len(l['text']) > max:
                    max = len(l['text'])
            
            for l in f2:
                l = json.loads(l)
                if len(l['text']) > max:
                    max = len(l['text'])
            
            for l in f3:
                l = json.loads(l)
                if len(l['text']) > max:
                    max = len(l['text'])
            
            print(max)
