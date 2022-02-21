from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from keras.utils import plot_model

from config import config_path, checkpoint_path, learning_rate
from path import *
from utils.backend import K, batch_gather
from utils.layers import LayerNormalization
from utils.layers import Loss
from utils.models import build_transformer_model
from load_schema import *


def get_model(predicate2id):
    # 补充输入
    subject_labels = Input(shape = (None, 2), name = 'Subject-Labels')
    subject_ids = Input(shape = (2,), name = 'Subject-Ids')
    object_labels = Input(shape = (None, len(predicate2id), 2), name = 'Object-Labels')
    
    # 加载预训练模型
    bert = build_transformer_model(
        config_path = config_path,
        checkpoint_path = checkpoint_path,
        return_keras_model = False,
        model = MODEL_TYPE
    )
    
    # 预测subject
    output = Dense(
        units = 2, activation = 'sigmoid', kernel_initializer = bert.initializer
    )(bert.model.output)
    subject_preds = Lambda(lambda x: x ** 2)(output)
    # 我们引入了Bert作为编码器，然后得到了编码序列bert.model.output，然后直接接一个输出维度是2的Dense来预测首尾
    
    subject_model = Model(bert.model.inputs, subject_preds)
    
    # 传入subject，预测object
    
    # 为了取Transformer-11-FeedForward-Add做Conditional Layer Normalization
    # 做Conditional Layer Normalization就不需要Transformer-11-FeedForward-Norm了
    output = bert.model.layers[-2].get_output_at(-1)
    # 把传入的s的首尾对应的编码向量拿出来,直接加到编码向量序列t中去
    subject = Lambda(extract_subject)([output, subject_ids])
    # 通过Conditional Layer Normalization将subject融入到object的预测中
    output = LayerNormalization(conditional = True)([output, subject])
    output = Dense(
        units = len(predicate2id) * 2,
        activation = 'sigmoid',
        kernel_initializer = bert.initializer
    )(output)
    output = Lambda(lambda x: x ** 4)(output)
    object_preds = Reshape((-1, len(predicate2id), 2))(output)
    
    object_model = Model(bert.model.inputs + [subject_ids], object_preds)
    
    subject_preds, object_preds = TotalLoss([2, 3])([
        subject_labels, object_labels, subject_preds, object_preds,
        bert.model.output
    ])
    
    # 训练模型
    train_model = Model(
        bert.model.inputs + [subject_labels, subject_ids, object_labels],
        [subject_preds, object_preds]
    )
    
    plot_model(train_model, to_file = fig_path + '/model.png', show_shapes = True)
    train_model.summary()
    
    return subject_model, object_model, train_model


def extract_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    把预测的subject对应的embedding与bert输出的hidden-states 进行拼接
    """
    output, subject_ids = inputs  # output:(?, ?, 312)    subject_ids:(?, 2)
    # Embedding - Token 出来的每个字都是312维
    # batch_gather
    # seq = [1,2,3,4,5,6]
    # idxs = [2,3,4]
    # a = K.tf.batch_gather(seq, idxs)
    # [3 4 5]
    start = batch_gather(output, subject_ids[:, :1])  # 取头指针 (?, 1, 312)
    end = batch_gather(output, subject_ids[:, 1:])  # 取尾指针 (?, 1, 312)
    subject = K.concatenate([start, end], 2)  # (?, 1, 624)
    return subject[:, 0]  # (?, 624)


class TotalLoss(Loss):
    """subject_loss与object_loss之和，都是二分类交叉熵
    """
    
    def compute_loss(self, inputs, mask = None):
        subject_labels, object_labels = inputs[:2]
        subject_preds, object_preds, _ = inputs[2:]
        if mask[4] is None:
            mask = 1.0
        else:
            mask = K.cast(mask[4], K.floatx())
        # sujuect部分loss
        subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
        subject_loss = K.mean(subject_loss, 2)
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
        # object部分loss
        object_loss = K.binary_crossentropy(object_labels, object_preds)
        object_loss = K.sum(K.mean(object_loss, 3), 2)
        object_loss = K.sum(object_loss * mask) / K.sum(mask)
        # 总的loss
        return subject_loss + object_loss


subject_model, object_model, train_model = get_model(predicate2id)
