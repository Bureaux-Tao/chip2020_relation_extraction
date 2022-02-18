import json

from config import maxlen
from dataloader import tokenizer
from model import get_model
from path import schema_path
from schemaloader import id2predicate
from utils.snippets import open, to_array
import numpy as np


def predict(text, model_path, threshold = 0):
    model = get_model()
    model.load_weights(model_path)
    tokens = tokenizer.tokenize(text, maxlen = maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen = maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    outputs = model.predict([token_ids, segment_ids])
    outputs = [o[0] for o in outputs]
    # 抽取subject和object
    subjects, objects = set(), set()
    outputs[0][:, [0, -1]] -= np.inf
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    # 识别对应的predicate
    spoes = set()
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)
            for p in ps:
                spoes.add((
                    text[mapping[sh][0]:mapping[st][-1] + 1], id2predicate[p],
                    text[mapping[oh][0]:mapping[ot][-1] + 1]
                ))
    
    relation_list = list(spoes)
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


if __name__ == '__main__':
    text = "妊娠胆汁淤积@[HELLP 综合征] ### 急性妊娠期脂肪肝 体征/症状 检查 体征/症状 患者感觉不适，常见表现为全身乏力、恶心，很多具有先兆子痫、凝血功能异常和肾脏损伤的典型症状。妊娠胆汁淤积@肝脏活检见脂肪浸润，但一般诊断过程中极少进行活检。妊娠胆汁淤积@[HELLP 综合征] ### 急性妊娠期脂肪肝 体征/症状 检查 体征/症状 患者感觉不适，常见表现为全身乏力、恶心，很多具有先兆子痫、凝血功能异常和肾脏损伤的典型症状。妊娠胆汁淤积@肝脏活检见脂肪浸润，但一般诊断过程中极少进行活检。"
    # text = "第三节 室间隔缺损 室间隔缺损（ventricular septal defect，VSD）是最常见的先天性心血管畸形，可占先心病人的20%。 有中至大型左向右分流，产生心力衰竭的婴儿，当可能出现缺损部分或完全自然关闭时，也可最初以药物治疗：①利尿剂降低心脏负荷和体循环静脉的充血状况。"
    # text = "B族链球菌感染@脓毒症 * 一线疗法：青霉素或氨苄西林。B族链球菌感染@ * 青霉素过敏患者：二代或三代头孢菌素（可能适用，具体取决于过敏反应类型）或者万古霉素。"
    print(json.dumps(predict(text = text,
                             model_path = "./weights/gplinker_roformer_best.h5"), ensure_ascii = False))
