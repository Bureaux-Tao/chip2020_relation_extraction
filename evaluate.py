import copy
import json

import pandas as pd
from tqdm import tqdm

from dataloader import SPO, extract_spoes, load_data
from model import model, get_model
from path import proj_path
from schemaloader import predicate2id


def evaluate(data, epoch, silent = False):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(proj_path + "/data/pred/val_ep" + str(epoch + 1) + ".json", 'w', encoding = 'utf-8')
    if not silent:
        pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in extract_spoes(d['text'], model)])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        if not silent:
            pbar.update()
            pbar.set_description(
                'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            )
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
            ensure_ascii = False,
            indent = 4)
        f.write(s + '\n')
    if not silent:
        pbar.close()
    f.close()
    return f1, precision, recall


def evaluate_each_relations_f1(model_path, dataset_path):
    """评估函数，计算f1、precision、recall
    """
    model = get_model()
    model.load_weights(model_path)
    data = load_data(dataset_path)
    
    predicate = {}
    for i in predicate2id:
        predicate[str(i)] = {'TP': 1e-10, 'TP+FP': 1e-10, 'TP+FN': 1e-10}
    
    pbar = tqdm(data, ncols = 100)
    for d in pbar:
        for i in predicate:
            R = set([SPO(spo) for spo in extract_spoes(d['text'], model)])
            R_labeled = set()
            for s, p, o in R:
                if p == i:
                    R_labeled.add((s, p, o))
            T = set([SPO(spo) for spo in d['spo_list']])
            T_labeled = set()
            for s, p, o in T:
                if p == i:
                    T_labeled.add((s, p, o))
            predicate[i]["TP"] += len(R_labeled & T_labeled)
            predicate[i]["TP+FP"] += len(R_labeled)
            predicate[i]["TP+FN"] += len(T_labeled)
        
        pbar.update()
        pbar.set_description("Evaluating F1 of each Categories")
    for i in predicate:
        predicate[i]["precision"] = round(predicate[i]["TP"] / predicate[i]["TP+FP"], 4)
        predicate[i]["recall"] = round(predicate[i]["TP"] / predicate[i]["TP+FN"], 4)
        predicate[i]["f1"] = round(2 * predicate[i]["TP"] / (predicate[i]["TP+FP"] + predicate[i]["TP+FN"]), 4)
        predicate[i]["TP"] = int(predicate[i]["TP"])
        predicate[i]["TP+FP"] = int(predicate[i]["TP+FP"])
        predicate[i]["TP+FN"] = int(predicate[i]["TP+FN"])
    pbar.close()
    return predicate


if __name__ == '__main__':
    result = evaluate_each_relations_f1(model_path = "./weights/gplinker_roformer_v2_base_best.h5",
                                        dataset_path = "./data/chip2020/val_data.json")
    df = pd.DataFrame(result)
    df = df.T
    df[["TP", "TP+FP", "TP+FN"]] = df[["TP", "TP+FP", "TP+FN"]].astype(int)
    # 设置value的显示长度为200，默认为50
    pd.set_option('max_colwidth', 200)
    # 显示所有列，把行显示设置成最大
    pd.set_option('display.max_columns', None)
    # 显示所有行，把列显示设置成最大
    pd.set_option('display.max_rows', None)
    df.to_csv("./report/predicate_f1_base.csv")
    print(df)
