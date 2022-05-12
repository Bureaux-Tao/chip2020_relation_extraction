import os

event_type = "gplinker"

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前地址
proj_path = current_dir

weights_path = proj_path + "/weights"
f1_report_path = proj_path + "/report/f1.csv"
log_path = proj_path + "/log/train_log.csv"
fig_path = proj_path + "/images"
label_dict_path = proj_path + "/data/%s_catagory.pkl" % event_type
categories_f1_path = proj_path + "/report/categories_f1.csv"

# KE
train_file_path = proj_path + "/data/chip2020/train_data.json"
test_file_path = proj_path + "/data/chip2020/test_data.json"
val_file_path = proj_path + "/data/chip2020/val_data.json"
schema_path = proj_path + "/data/chip2020/53_schemas.json"

# Model Config
MODEL_TYPE = 'roformer_v2'

BASE_MODEL_DIR = proj_path + "/chinese_roformer-v2-char_L-12_H-768_A-12"
BASE_CONFIG_NAME = proj_path + "/chinese_roformer-v2-char_L-12_H-768_A-12/bert_config.json"
BASE_CKPT_NAME = proj_path + "/chinese_roformer-v2-char_L-12_H-768_A-12/bert_model.ckpt"
