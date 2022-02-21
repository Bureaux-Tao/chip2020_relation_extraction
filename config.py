from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR

maxlen = 300
batch_size = 128
learning_rate = 1e-4

# bert配置
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)
