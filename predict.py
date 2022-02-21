import json

import numpy as np
from casrel import get_model
from config import maxlen
from data_process import tokenizer
from load_schema import predicate2id, id2predicate
from path import schema_path
from utils.snippets import open, to_array


def predict(text, model_path):
    subject_model, object_model, train_model = get_model(predicate2id)
    train_model.load_weights(model_path)
    tokens = tokenizer.tokenize(text, maxlen = maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen = maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    # 抽取subject
    subject_preds = subject_model.predict([token_ids, segment_ids])
    subject_preds[:, [0, -1]] *= 0
    start = np.where(subject_preds[0, :, 0] > 0.6)[0]
    end = np.where(subject_preds[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat(token_ids, len(subjects), 0)
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        subjects = np.array(subjects)
        # 传入subject，抽取object和predicate
        object_preds = object_model.predict([token_ids, segment_ids, subjects])
        object_preds[:, [0, -1]] *= 0
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.6)
            end = np.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((mapping[subject[0]][0],
                              mapping[subject[1]][-1]), predicate1,
                             (mapping[_start][0], mapping[_end][-1]))
                        )
                        break
        relation_list = [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                         for s, p, o, in spoes]
        dict_list = []
        for s, p, o in relation_list:
            i = 0
            is_subject_exists = False
            while i < len(dict_list):
                if dict_list[i]["subject"] == s:
                    is_subject_exists = True
                    if not (p in dict_list[i]["attributes"].keys()):
                        dict_list[i]["attributes"][p] = [o]
                    else:
                        dict_list[i]["attributes"][p].append(o)
                i += 1
            if not is_subject_exists:
                dict_list.append({"subject": s, "attributes": {p: [o]}})
        
        schemas = []
        with open(schema_path, encoding = 'utf-8') as f:
            for l in f:
                schemas.append(json.loads(l))
        five_tuple_list = []
        for i in relation_list:
            for j in schemas:
                if i[1] == j["predicate"]:
                    five_tuple_list.append(
                        {"subject": i[0], "subject_type": j["subject_type"], "predicate": i[1], "object": i[2],
                         "object_type": j["object_type"]})
        
        return {"text": text,
                "spo_list": five_tuple_list,
                "standarlization_list": dict_list}
    
    else:
        return []


if __name__ == '__main__':
    text = "营养性巨幼细胞性贫血（nutritional megaloblastic anemia)是由于维生素B12和（或）叶酸缺乏所致的一种大细胞性贫血。因使用抗叶酸代谢药物而致病者，可用亚叶酸钙（calc leucovorin)治疗。"
    # text = "第三节 室间隔缺损 室间隔缺损（ventricular septal defect，VSD）是最常见的先天性心血管畸形，可占先心病人的20%。 有中至大型左向右分流，产生心力衰竭的婴儿，当可能出现缺损部分或完全自然关闭时，也可最初以药物治疗：①利尿剂降低心脏负荷和体循环静脉的充血状况。"
    # text = "B族链球菌感染@脓毒症 * 一线疗法：青霉素或氨苄西林。B族链球菌感染@ * 青霉素过敏患者：二代或三代头孢菌素（可能适用，具体取决于过敏反应类型）或者万古霉素。"
    print(json.dumps(predict(text = text,
                             model_path = "./weights/chip2020_roformer_best.h5"), ensure_ascii = False))
