import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd
import keras

from utils_for_clsfy_skin_cancer import encode_age, encode_site, encode_sex
from utils_for_clsfy_skin_cancer import parse_label
from utils_for_clsfy_skin_cancer import preprocess_train_data, preprocess_validation_data
from utils_for_clsfy_skin_cancer import create_model
from utils_for_clsfy_skin_cancer import dir
from utils_for_clsfy_skin_cancer import train_data_fn, train_label_fn, train_input_dir
from utils_for_clsfy_skin_cancer import train_cache_dir, validation_cache_dir


# 超参数
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32


# 确保缓存路径
if not os.path.exists(train_cache_dir):
     os.makedirs(train_cache_dir)

if not os.path.exists(validation_cache_dir):
     os.makedirs(validation_cache_dir)


# 清理锁文件
train_lockfile_fp = os.path.join(train_cache_dir, '_0.lockfile')
if os.path.exists(train_lockfile_fp):
     os.remove(train_lockfile_fp)

validation_lockfile_fp = os.path.join(validation_cache_dir, '_0.lockfile')
if os.path.exists(validation_lockfile_fp):
     os.remove(validation_lockfile_fp)


# 读取数据
df_data = pd.read_csv(os.path.join(dir, train_data_fn))
df_label = pd.read_csv(os.path.join(dir, train_label_fn))


# 提取数据特征
train_image_fp = df_data['image']\
    .apply(lambda x: os.path.join(dir, train_input_dir, x + '.jpg')).values  # 病变图像路径
train_age = encode_age(df_data)  # 年龄
train_site = encode_site(df_data)  # 病变部位
train_sex = encode_sex(df_data)  # 性别
train_label = parse_label(df_label)  # 病种


# 划分训练集 验证集
ds = tf.data.Dataset.from_tensor_slices(
    (train_image_fp, train_age, train_site, train_sex, train_label)
)

ds_size = len(ds)
train_size = int(ds_size * 0.9)

ds_train = ds.take(train_size)
ds_validation = ds.skip(train_size)


# 预处理
ds_train = ds_train.map(preprocess_train_data, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache(train_cache_dir)
ds_train = ds_train.shuffle(5000)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_validation = ds_validation.map(preprocess_validation_data, num_parallel_calls=AUTOTUNE)
ds_validation = ds_validation.cache(validation_cache_dir)
ds_validation = ds_validation.batch(BATCH_SIZE)
ds_validation = ds_validation.prefetch(AUTOTUNE)


# 初始化模型
model = create_model()


# 回调
# 调整学习率
def scheduler(epoch, lr):
     if epoch > 20:
          return lr * 0.99
lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)

# 精度控制
class CustomCallback(keras.callbacks.Callback):
     def on_epoch_end(self, epoch, logs=None):
          if logs.get['accuracy'] > 0.9:
               self.model.stop_training = True


# 配置模型的学习过程,定义模型在训练时使用的损失函数 优化器和评估指标
model.compile(
     loss=keras.losses.SparseCategoricalCrossentropy(),
     optimizer=keras.optimizers.Adam(learning_rate=0.01),
     metrics=['accuracy']
)


# 训练数据
model.fit(
     ds_train,
     epochs=50,
     validation_data=ds_validation,
     callbacks=[lr_scheduler, CustomCallback],
     verbose=2
)