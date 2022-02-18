from config import config_path, checkpoint_path
from path import MODEL_TYPE
from schemaloader import predicate2id
from utils.backend import keras, K
from utils.backend import sparse_multilabel_categorical_crossentropy
from utils.layers import GlobalPointer
from utils.models import build_transformer_model


def globalpointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    shape = K.shape(y_pred)
    y_true = y_true[..., 0] * K.cast(shape[2], K.floatx()) + y_true[..., 1]
    y_pred = K.reshape(y_pred, (shape[0], -1, K.prod(shape[2:])))
    loss = sparse_multilabel_categorical_crossentropy(y_true, y_pred, True)
    return K.mean(K.sum(loss, axis = 1))


def get_model():
    # 加载预训练模型
    base = build_transformer_model(
        config_path = config_path,
        checkpoint_path = checkpoint_path,
        model = MODEL_TYPE,
        return_keras_model = False
    )
    
    # 预测结果
    entity_output = GlobalPointer(heads = 2, head_size = 64)(base.model.output)
    head_output = GlobalPointer(
        heads = len(predicate2id), head_size = 64, RoPE = False, tril_mask = False
    )(base.model.output)
    tail_output = GlobalPointer(
        heads = len(predicate2id), head_size = 64, RoPE = False, tril_mask = False
    )(base.model.output)
    outputs = [entity_output, head_output, tail_output]
    
    model = keras.models.Model(base.model.inputs, outputs)
    return model


model = get_model()
