import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.ERROR)

import tensorflow as tf
import pandas as pd

from utils_for_clsfy_skin_cancer import (
    configure_GPU,
    directory, data_dir,
    test_data_fn, test_input_dir,
    saved_model_dir,
    encode_age, encode_site, encode_sex,
    SkinCancerModel
)


# 配置GPU
configure_GPU()


# 超参数
TEST_BATCH_SIZE = 128


# 读取文件
test_data = pd.read_csv(os.path.join(directory, data_dir, test_data_fn))


# 提取数据特征
test_image_fp = test_data['image']\
    .apply(lambda x: os.path.join(directory, data_dir, test_input_dir, x + '.jpg')).values  # 病变图像路径
test_age = encode_age(test_data)  # 年龄
test_site = encode_site(test_data)  # 病变部位
test_sex = encode_sex(test_data)  # 性别


# 加载模型
model = SkinCancerModel()
model.load_model(os.path.join(directory, saved_model_dir, 'completed_model.keras'))


# 预测标签值
model.predict(
    [test_image_fp, test_age, test_site, test_sex],
    test_batch_size=TEST_BATCH_SIZE
)