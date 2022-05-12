from config import config_path, checkpoint_path
from path import MODEL_TYPE
from schemaloader import predicate2id
from utils.backend import keras, K
from utils.backend import sparse_multilabel_categorical_crossentropy
from utils.layers import GlobalPointer
from utils.models import build_transformer_model


class SetLearningRate:
    """层的一个包装，用来设置当前层的学习率
    """
    
    def __init__(self, layer, lamb, is_ada = False):
        self.layer = layer
        self.lamb = lamb  # 学习率比例
        self.is_ada = is_ada  # 是否自适应学习率优化器
    
    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma',
                    'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb  # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb ** 0.5  # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight) / lamb)  # 更改初始化
                setattr(self.layer, key, weight * lamb)  # 按比例替换
        return self.layer(inputs)


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
    entity_output = SetLearningRate(GlobalPointer(heads = 2, head_size = 64, kernel_initializer = "he_normal"),
                                    10, True)(base.model.output)
    head_output = SetLearningRate(GlobalPointer(
        heads = len(predicate2id), head_size = 64, RoPE = False, tril_mask = False, kernel_initializer = "he_normal"
    ), 10, True)(base.model.output)
    tail_output = SetLearningRate(GlobalPointer(
        heads = len(predicate2id), head_size = 64, RoPE = False, tril_mask = False, kernel_initializer = "he_normal"
    ), 10, True)(base.model.output)
    outputs = [entity_output, head_output, tail_output]
    
    model = keras.models.Model(base.model.inputs, outputs)
    return model


model = get_model()
