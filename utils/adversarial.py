import numpy as np
from utils.backend import keras, search_layer, K


def adversarial_training(model, embedding_name, epsilon = 1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数
    
    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')
    
    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor
    
    # 封装为函数
    inputs = (
            model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs = inputs,
        outputs = [gradients],
        name = 'embedding_gradients',
    )  # 封装为函数
    
    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads ** 2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs
    
    model.train_function = train_function  # 覆盖原训练函数


def virtual_adversarial_training(
        model, embedding_name, epsilon = 1, xi = 10, iters = 1
):
    """给模型添加虚拟对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数
    
    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')
    
    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor
    
    # 封装为函数
    inputs = (
            model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    model_outputs = K.function(
        inputs = inputs,
        outputs = model.outputs,
        name = 'model_outputs',
    )  # 模型输出函数
    embedding_gradients = K.function(
        inputs = inputs,
        outputs = [gradients],
        name = 'embedding_gradients',
    )  # 模型梯度函数
    
    def l2_normalize(x):
        return x / (np.sqrt((x ** 2).sum()) + 1e-8)
    
    def train_function(inputs):  # 重新定义训练函数
        outputs = model_outputs(inputs)
        inputs = inputs[:2] + outputs + inputs[3:]
        delta1, delta2 = 0.0, np.random.randn(*K.int_shape(embeddings))
        for _ in range(iters):  # 迭代求扰动
            delta2 = xi * l2_normalize(delta2)
            K.set_value(embeddings, K.eval(embeddings) - delta1 + delta2)
            delta1 = delta2
            delta2 = embedding_gradients(inputs)[0]  # Embedding梯度
        delta2 = epsilon * l2_normalize(delta2)
        K.set_value(embeddings, K.eval(embeddings) - delta1 + delta2)
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta2)  # 删除扰动
        return outputs
    
    model.train_function = train_function  # 覆盖原训练函数
