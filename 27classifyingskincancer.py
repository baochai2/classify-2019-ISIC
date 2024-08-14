import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd
import keras

from utils_for_clsfy_skin_cancer import directory
from utils_for_clsfy_skin_cancer import train_data_fn, train_label_fn, train_input_dir
from utils_for_clsfy_skin_cancer import train_cache_dir, validation_cache_dir
from utils_for_clsfy_skin_cancer import encode_age, encode_site, encode_sex
from utils_for_clsfy_skin_cancer import parse_label
from utils_for_clsfy_skin_cancer import create_model
from utils_for_clsfy_skin_cancer import lr_scheduler, CustomCallback
from utils_for_clsfy_skin_cancer import train_model_in_batches


# 配置GPU
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices[0])
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# 超参数
TRAIN_VALIDATION_RATE = 0.9
AUTOTUNE = tf.data.experimental.AUTOTUNE
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 50


# 确保缓存路径
for cache in [train_cache_dir, validation_cache_dir]:
     if not os.path.exists(cache):
          os.makedirs(cache)


# 清理锁文件
train_lockfile_fp = os.path.join(train_cache_dir, '_0.lockfile')
validation_lockfile_fp = os.path.join(validation_cache_dir, '_0.lockfile')
for lockfile in [train_lockfile_fp, validation_lockfile_fp]:
     if os.path.exists(lockfile):
          os.remove(lockfile)


# 读取数据
df_data = pd.read_csv(os.path.join(directory, train_data_fn))
df_label = pd.read_csv(os.path.join(directory, train_label_fn))


# 提取数据特征
train_image_fp = df_data['image']\
    .apply(lambda x: os.path.join(directory, train_input_dir, x + '.jpg')).values  # 病变图像路径
train_age = encode_age(df_data)  # 年龄
train_site = encode_site(df_data)  # 病变部位
train_sex = encode_sex(df_data)  # 性别
train_label = parse_label(df_label)  # 病种


# 划分训练集 验证集
ds_size = len(train_image_fp)
train_size = int(ds_size * TRAIN_VALIDATION_RATE)

train_image_fp, validation_image_fp = train_image_fp[:train_size], train_image_fp[train_size:]
train_age, validation_age = train_age[:train_size], train_age[train_size:]
train_site, validation_site = train_site[:train_size], train_site[train_size:]
train_sex, validation_sex = train_sex[:train_size], train_sex[train_size:]
train_label, validation_label = train_label[:train_size], train_label[train_size:]


# 初始化模型
model = create_model()


# 配置模型的学习过程,定义模型在训练时使用的损失函数 优化器和评估指标
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizer_v2.adam.Adam(learning_rate=LEARNING_RATE),
    metrics=['accuracy']
)


# 训练数据
train_model_in_batches(
    model,
    [train_image_fp, train_age, train_site, train_sex],
    train_label,
    [validation_image_fp, validation_age, train_site, train_sex],
    validation_label,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[lr_scheduler, CustomCallback()]
)