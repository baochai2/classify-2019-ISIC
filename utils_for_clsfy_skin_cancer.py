import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import keras
import time

from keras import layers, regularizers


# 超参数
HEIGHT = 400
WIDTH = 400
CHANNEL = 3
AUTOTUNE = tf.data.experimental.AUTOTUNE
VERBOSE_MODE = 0


# 数据路径
dir = 'skin_cancer/'

train_label_fn = 'ISIC_2019_Training_GroundTruth.csv'
train_input_dir = 'ISIC_2019_Training_Input'

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


# 解析出文件的标签
def parse_label(dataframe):
    label = np.array([], dtype='int32')
    for row in np.array(dataframe[label_names]):  # 按行遍历每一行的病种
        idx = np.where(row == 1.0)[0][0]  # 按行遍历每一行的病种
        label = np.append(label, idx)
    return label


# 预处理
# 将读取到的图像名转化为张量
def read_image(image, label):
    image = tf.io.read_file(image)  # 读取文件

    image = tf.image.decode_jpeg(image, channels=CHANNEL)
        # image数据是由height*width个channel张量组成
    assert image.shape[-1] == CHANNEL, f"图形通道数不匹配"
    return image, label

# 标准化
def normalize_image(image, label):
    image = tf.image.resize(image, (HEIGHT, WIDTH))  # 可调整插值方法method
    image.set_shape([HEIGHT, WIDTH, CHANNEL])
    return tf.cast(image, dtype=tf.float32) / 255.0, label

# 数据增强
def augment(image, label):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_saturation(image, 5, 10)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label

def ensure_data(image, label):
    assert image.dtype == tf.float32, f"Image dtype is not tf.float32, but {image.dtype}"
    assert image.shape == (HEIGHT, WIDTH, CHANNEL),\
        f"Image shape is not ({HEIGHT}, {WIDTH}, {CHANNEL}), but {image.shape}"

def build_inputs(image, label):
    inputs = {
        'image_input': image
    }  # 传入字典的键应该与模型输入层的name相同
    return inputs, label

def preprocess_train_data(image, label):
    image, label  = read_image(image, label)
    image, label = normalize_image(image, label)
    image, label = augment(image, label)
    ensure_data(image, label)
    inputs, label = build_inputs(image, label)
    return inputs, label

def preprocess_validation_data(image, label):
    image, label  = read_image(image, label)
    image, label = normalize_image(image, label)
    ensure_data(image, label)
    inputs, label = build_inputs(image, label)
    return inputs, label


# 构建模型
def create_model():

    # 图像输入
    image_inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNEL), name='image_input')
    x = layers.Conv2D(
        HEIGHT / 16, 7, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(image_inputs)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(
        HEIGHT / 8, 7, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(
        HEIGHT / 8, 7, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(
        HEIGHT / 4, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(
        HEIGHT / 4, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(
        HEIGHT / 2, 5, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(
        HEIGHT / 2, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.Flatten()(x)
    x = layers.Dense(HEIGHT / 4, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dense(HEIGHT / 8, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(HEIGHT / 16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    outputs = layers.Dense(len(label_names), activation='softmax')(x)
    model = keras.Model(inputs={'image_input': image_inputs}, outputs=outputs)
    return model


# 回调
# 调整学习率
def scheduler(epoch, lr):
     if epoch > 3:
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
        epochs,
        callbacks
):
    num_batches = len(train_datas) // batch_size

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
            batch_data = train_datas[batch_start: batch_end]
            batch_label = train_labels[batch_start: batch_end]

            # 预处理
            batch_ds = tf.data.Dataset.from_tensor_slices((batch_data, batch_label))
            batch_ds = batch_ds.map(preprocess_train_data, num_parallel_calls=AUTOTUNE)
            batch_ds = batch_ds.prefetch(AUTOTUNE)

            # 转为numpy数组
            batch_inputs, batch_targets = [], []
            for inputs, targets in batch_ds:
                batch_inputs.append(inputs['image_input'].numpy())
                batch_targets.append(targets.numpy())
            
            batch_inputs = np.array(batch_inputs)
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
        val_inputs, val_targets = [], []
        for val_data, val_label in zip(validation_datas, validation_labels):
            val_data, val_label = preprocess_validation_data(val_data, val_label)
            val_inputs.append(val_data['image_input'].numpy())
            val_targets.append(val_label)

        val_inputs = np.array(val_inputs)
        val_targets = np.array(val_targets)

        # 计算验证集的loss和accuracy
        val_loss, val_accuracy = model.evaluate(val_inputs, val_targets, verbose=VERBOSE_MODE)
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