import json
import numpy as np

# 建立分词器
from config import dict_path, maxlen
from schemaloader import predicate2id, id2predicate
from utils.snippets import DataGenerator, sequence_padding
from utils.tokenizers import Tokenizer
from utils.snippets import open, to_array

tokenizer = Tokenizer(dict_path, do_lower_case = True)


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o)]}
    """
    D = []
    with open(filename, encoding = 'utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text'],
                'spo_list': [(spo['subject'], spo['predicate'], spo['object'])
                             for spo in l['spo_list']]
            })
    return D


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    
    def __iter__(self, random = False):
        batch_token_ids, batch_segment_ids = [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(d['text'], maxlen = maxlen)
            # 整理三元组 {(s, o, p)}
            spoes = set()
            for s, p, o in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                sh = search(s, token_ids)
                oh = search(o, token_ids)
                if sh != -1 and oh != -1:
                    spoes.add((sh, sh + len(s) - 1, p, oh, oh + len(o) - 1))
            # 构建标签
            entity_labels = [set() for _ in range(2)]
            head_labels = [set() for _ in range(len(predicate2id))]
            tail_labels = [set() for _ in range(len(predicate2id))]
            for sh, st, p, oh, ot in spoes:
                entity_labels[0].add((sh, st))
                entity_labels[1].add((oh, ot))
                head_labels[p].add((sh, oh))
                tail_labels[p].add((st, ot))
            for label in entity_labels + head_labels + tail_labels:
                if not label:  # 至少要有一个标签
                    label.add((0, 0))  # 如果没有则用0填充
            entity_labels = sequence_padding([list(l) for l in entity_labels])
            head_labels = sequence_padding([list(l) for l in head_labels])
            tail_labels = sequence_padding([list(l) for l in tail_labels])
            # 构建batch
            L = len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_entity_labels = sequence_padding(
                    batch_entity_labels, seq_dims = 2
                )
                batch_head_labels = sequence_padding(
                    batch_head_labels, seq_dims = 2
                )
                batch_tail_labels = sequence_padding(
                    batch_tail_labels, seq_dims = 2
                )
                yield [batch_token_ids, batch_segment_ids], [
                    batch_entity_labels, batch_head_labels, batch_tail_labels
                ]
                batch_token_ids, batch_segment_ids = [], []
                batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []


def extract_spoes(text, model,threshold = 0):
    """抽取输入text所包含的三元组
    """
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
    return list(spoes)


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )
    
    def __hash__(self):
        return self.spox.__hash__()
    
    def __eq__(self, spo):
        return self.spox == spo.spox
