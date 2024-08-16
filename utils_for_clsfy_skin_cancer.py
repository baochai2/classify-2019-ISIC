import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import time

from keras import layers, regularizers


# 超参数
HEIGHT = 300
WIDTH = 300
CHANNEL = 3
NUM_KERNELS = 400
NUM_DENSE_UNITS = 128
VERBOSE_MODE = 0


# 数据路径
directory = 'skin_cancer/'

train_data_fn = 'ISIC_2019_Training_Metadata.csv'
train_label_fn = 'ISIC_2019_Training_GroundTruth.csv'
train_input_dir = 'ISIC_2019_Training_Input'

test_data_fn = 'ISIC_2019_Test_Metadata.csv'
test_input_dir = 'ISIC_2019_Test_Input'

train_cache_dir = 'D:/all_cache/code_cache/train_cache/'
validation_cache_dir = 'D:/all_cache/code_cache/validation_cache/'
test_cache_dir = 'D:/all_cache/code_cache/test_cache/'


# 类别名称
# 病变部位类别
site_names = [
    'NaN',
    'oral/genital',
    'lower extremity',
    'palms/soles',
    'lateral torso',
    'posterior torso',
    'head/neck',
    'anterior torso',
    'upper extremity'
]

# 性别
sex_names=[
    'NaN',
    'male',
    'female'
]

# 病种类别
label_names = [
    'MEL',
    'NV',
    'BCC',
    'AK',
    'BKL',
    'DF',
    'VASC',
    'SCC',
    'UNK'
]


# 确保缓存路径
def ensure_cache_dir(all_cache_dir):
    for cache_dir in all_cache_dir:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)


# 清理锁文件
def clean_lockfile(all_lockfile_fp):
    for lockfile in all_lockfile_fp:
        if os.path.exists(lockfile):
            os.remove(lockfile)


# 编码未处理的数据特征
def encode_age(dataframe):
    age = np.array([], dtype='float64')  # 初始化存储特征的列表
    for row in np.array(dataframe['age_approx']):
        if pd.isna(row): row = 0  # 如果row为缺失值NaN
        age = np.append(age, row / 100)
    return age

def encode_site(dataframe):
    site = np.array([], dtype='float64')
    for row in np.array(dataframe['anatom_site_general']):
        idx = 0 if pd.isna(row) else site_names.index(row)
        site = np.append(site, idx / len(site_names))
    return site

def encode_sex(dataframe):
    sex = np.array([], dtype='float64')
    for row in np.array(dataframe['sex']):
        if pd.isna(row):
            idx = 0
        elif row == 'male':
            idx = 1
        else:
            idx = 2
        sex = np.append(sex, idx / len(sex_names))
    return sex


# 解析出文件的标签
def parse_label(dataframe):
    label = np.array([np.argmax(row) for row in dataframe[label_names].values], dtype='int32')
    return label


# 预处理
# 将读取到的图像名转化为张量
def read_image(image):
    image = tf.io.read_file(image)  # 读取文件

    image = tf.image.decode_jpeg(image, channels=CHANNEL)
        # image数据是由height*width个channel张量组成
    assert image.shape[-1] == CHANNEL, f"图形通道数不匹配"
    return image

# 标准化
def normalize_image(image):
    image = tf.image.resize(image, (HEIGHT, WIDTH))  # 可调整插值方法method
    image.set_shape([HEIGHT, WIDTH, CHANNEL])
    return tf.cast(image, dtype=tf.float64) / 255.0

# 数据增强
def augment(image):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_saturation(image, 5, 10)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image

# 融合数据
def merge(image, age, site, sex):
    age = tf.broadcast_to(tf.reshape(age, [1, 1, 1]), [HEIGHT, WIDTH // 4, 1])
    site = tf.broadcast_to(tf.reshape(site, [1, 1, 1]), [HEIGHT, WIDTH // 2, 1])
    sex = tf.broadcast_to(tf.reshape(sex, [1, 1, 1]), [HEIGHT, WIDTH // 4, 1])

    data = tf.concat([age, site, sex], axis=-2)
    data = tf.concat([image, data], axis=-1)
    return data

# 确保数据格式
def ensure_data(data):
    assert data.dtype == tf.float64, f"data dtype is not tf.float64, but {data.dtype}"
    assert data.shape == (HEIGHT, WIDTH, CHANNEL + 1),\
        f"Data shape is not ({HEIGHT}, {WIDTH}, {CHANNEL + 1}), but {data.shape}"

# 构建输入
def build_inputs(data):
    inputs = {
        'data_input': data
    }  # 传入字典的键应该与模型输入层的name相同
    return inputs

def preprocess_train_data(data, label):
    image, age, site, sex = data
    image  = read_image(image)
    image = normalize_image(image)
    image = augment(image)
    data = merge(image, age, site, sex)
    ensure_data(data)
    inputs = build_inputs(data)
    return inputs, label

def preprocess_validation_data(data, label):
    image, age, site, sex = data
    image  = read_image(image)
    image = normalize_image(image)
    data = merge(image, age, site, sex)
    ensure_data(data)
    inputs = build_inputs(data)
    return inputs, label


# 构建模型
def create_model():

    # 输入
    data_inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNEL + 1), name='data_input')
    x = layers.Conv2D(
        NUM_KERNELS // 8, 7, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(data_inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS // 4, 7, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS // 4, 7, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS // 2, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS // 2, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(NUM_DENSE_UNITS, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dense(NUM_DENSE_UNITS // 2, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(NUM_DENSE_UNITS // 4, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)

    # 输出
    outputs = layers.Dense(len(label_names), activation='softmax')(x)
    model = keras.Model(inputs={'data_input': data_inputs}, outputs=outputs)
    return model


# 回调
# 调整学习率
def scheduler(epoch, lr):
     if epoch > 2:
          return lr * 0.99
     else:
          return lr
lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)

# 精度控制
class CustomCallback(keras.callbacks.Callback):
     def on_epoch_end(self, epoch, logs=None):
          if logs.get('accuracy') > 0.9:
               self.model.stop_training = True


# 手动分批次训练
def train_model_in_batches(
        model,
        train_datas,
        train_labels,
        validation_datas,
        validation_labels,
        batch_size,
        validation_batch_size,
        epochs,
        callbacks
):
    num_batches = len(train_labels) // batch_size
    num_val_batches = len(validation_labels) // validation_batch_size

    # 为每个回调类设定模型
    for callback in callbacks:
        callback.set_model(model)

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        start = time.time()

        # 调用学习率调整器
        for callback in callbacks:
            if isinstance(callback, keras.callbacks.LearningRateScheduler):
                callback.on_epoch_begin(epoch)

        epoch_loss, epoch_accuracy = 0, 0
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size

            # 获取当前批次的数据
            batch_data = tuple([data[batch_start: batch_end] for data in train_datas])
            batch_label = train_labels[batch_start: batch_end]

            # 预处理
            batch_datas, batch_labels = [], batch_label
            for image, age, site, sex in zip(*batch_data):
                processed_batch_datas, _ = \
                    preprocess_train_data((image, age, site, sex), batch_label)
                batch_datas.append(processed_batch_datas)

            # 转为numpy数组
            batch_inputs, batch_targets = [], []
            for inputs, targets in zip(batch_datas, batch_labels):
                batch_inputs.append(inputs['data_input'].numpy())
                batch_targets.append(targets)
            
            batch_inputs = np.array([inputs['data_input'].numpy() for inputs in batch_datas])
            batch_targets = np.array(batch_targets)

            # 训练数据
            batch_loss, batch_accuracy = model.train_on_batch(batch_inputs, batch_targets)

            # 累加loss和accuracy
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy

        # 计算每个epoch的loss和accuracy
        epoch_loss /= num_batches
        epoch_accuracy /= num_batches
        print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}')

        # 将验证集转为numpy数组
        val_loss, val_accuracy = 0, 0
        val_inputs, val_targets = [], []
        for i in range(num_val_batches):
            val_batch_start = i * validation_batch_size
            val_batch_end = (i + 1) * validation_batch_size

            val_batch_data = tuple([data[val_batch_start: val_batch_end] for data in validation_datas])
            val_batch_label = validation_labels[val_batch_start: val_batch_end]

            val_datas, val_labels = [], val_batch_label
            for image, age, site, sex in zip(*val_batch_data):
                processed_val_datas, _ = \
                    preprocess_validation_data((image, age, site, sex), val_batch_label)
                val_datas.append(processed_val_datas)

            val_inputs, val_targets = [], []
            for inputs, targets in zip(val_datas, val_labels):
                val_inputs.append(inputs['data_input'].numpy())
                val_targets.append(targets)

            val_inputs = np.array(val_inputs)
            val_targets = np.array(val_targets)

            val_batch_loss, val_batch_accuracy= \
                model.evaluate(val_inputs, val_targets, verbose=VERBOSE_MODE)
            
            val_loss += val_batch_loss
            val_accuracy += val_batch_accuracy

        # 计算验证集的loss和accuracy
        val_loss /= num_val_batches
        val_accuracy /= num_val_batches
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # 调用精度调整器
        for callback in callbacks:
            if isinstance(callback, keras.callbacks.Callback):
                callback.on_epoch_end(epoch, logs={'accuracy': epoch_accuracy})
        
        if model.stop_training == True:
            print('Stopping training early because of accuracy')
            break

        # 一个epoch的耗时
        end = time.time()
        epoch_duration = end - start
        print(f'Epoch {epoch + 1} takes {epoch_duration:.2f} seconds')