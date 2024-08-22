import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.ERROR)

import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import time

from keras import layers, regularizers


# 超参数
HEIGHT = 224
WIDTH = 224
CHANNEL = 3
NUM_KERNELS = 400
NUM_DENSE_UNITS = 128
LEARNING_RATE = 1e-3
VERBOSE_MODE = 0


# 数据路径
directory = 'skin_cancer/'
data_dir = 'data/'

train_data_fn = 'ISIC_2019_Training_Metadata.csv'
train_label_fn = 'ISIC_2019_Training_GroundTruth.csv'
train_input_dir = 'ISIC_2019_Training_Input'

test_data_fn = 'ISIC_2019_Test_Metadata.csv'
test_input_dir = 'ISIC_2019_Test_Input'

checkpoint_dir = 'checkpoint_skin_cancer/'
checkpoint_fn = 'best_model_weights.h5'

tensorboard_dir = 'tb_skin_cancer/'

saved_model_dir = 'saved_model/'


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


# 配置GPU
def configure_GPU():
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f'\n{physical_devices[0]}\n')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            print('\nNo GPU is available\n')
    except Exception as e:
        print(f'\nError configuring GPU: {e}\n')


# 确保路径
def ensure_dir(all_dir):
    for one_dir in all_dir:
        if not os.path.exists(one_dir):
            os.makedirs(one_dir)


# 编码未处理的数据特征
def encode_age(dataframe):
    age = np.array(dataframe['age_approx'].fillna(0), dtype='float64')  # 初始化存储特征的列表
    mean = np.mean(age)
    std = np.std(age)
    age = (age - mean) / std  # 中心化处理
    return age

def encode_site(dataframe):
    site = np.array([0 if pd.isna(row) else site_names.index(row) 
                     for row in dataframe['anatom_site_general']], dtype='float64')
    mean = np.mean(site)
    std = np.std(site)
    site = (site - mean) / std
    return site

def encode_sex(dataframe):
    sex = np.array([0 if pd.isna(row) else (1 if row == 'male' else 2) 
                    for row in dataframe['sex']], dtype='float64')
    mean = np.mean(sex)
    std = np.std(sex)
    sex = (sex - mean) / std
    return sex


# 解析出文件的标签
def parse_label(dataframe):
    label = np.array([np.argmax(row) for row in dataframe[label_names].values], dtype='int8')
    return label


# 预处理
# 将读取到的图像名转化为张量
def read_image(image):
    image = tf.io.read_file(image)  # 读取文件

    image = tf.image.decode_jpeg(image, channels=CHANNEL)
        # image数据是由height*width个channel张量组成
    assert image.shape[-1] == CHANNEL, f'图形通道数不匹配\n'
    return image

# 标准化
def normalize_image(image):
    image = tf.image.resize(image, (HEIGHT, WIDTH))  # 可调整插值方法method
    image.set_shape([HEIGHT, WIDTH, CHANNEL])
    image = tf.cast(image, dtype=tf.float64)

    # 中心化
    mean = tf.reduce_mean(image, [0, 1, 2])
    std = tf.math.reduce_std(image, [0, 1, 2])
    image = (image - mean) / std
    return image

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
    assert data.dtype == tf.float64, f'data dtype is not tf.float64, but {data.dtype}\n'
    assert data.shape == (HEIGHT, WIDTH, CHANNEL + 1),\
        f'Data shape is not ({HEIGHT}, {WIDTH}, {CHANNEL + 1}), but {data.shape}\n'

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

def preprocess_test_data(data, label):
    return preprocess_validation_data(data, label)


# 构建模型
def create_model():

    # 输入
    data_inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNEL + 1), name='data_input')
    x = layers.Conv2D(
        NUM_KERNELS // 8, 7, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(data_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS // 4, 7, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS // 4, 7, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS // 2, 5, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS // 2, 5, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS, 5, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS, 3, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        NUM_DENSE_UNITS, kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(
        NUM_DENSE_UNITS // 2, kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(
        NUM_DENSE_UNITS // 4, kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 输出
    outputs = layers.Dense(len(label_names), activation='softmax')(x)
    model = keras.Model(inputs={'data_input': data_inputs}, outputs=outputs)
    return model


# 回调
# 调整学习率
def scheduler(epoch, lr):
    if epoch % 10:
        return lr * 0.1
    else:
        return lr
lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)

# 精度控制
class AccuracyCallback(keras.callbacks.Callback):
     def on_epoch_end(self, epoch, logs=None):
          if logs.get('accuracy') > 0.9:
               self.model.stop_training = True

# 早停
class EarlyStoppingCallback:
    def __init__(self, monitor='val_loss', mode='min', min_delta=0.0001, patience=5):
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.wait = 0
        self.patience = patience

        if self.mode == 'min':
            self.best = float('inf')
        elif self.mode == 'max':
            self.best = -float('inf')
    
    def set_model(self, model):
        self.model = model
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)

        if monitor_value is None:
            print(f'Warning: Early stopping requires {self.monitor} availabe in logs.\n')
            return

        if self.mode == 'min':
            if monitor_value < self.best - self.min_delta:
                self.wait = 0
                self.best = monitor_value
            else:
                self.wait += 1
        elif self.mode == 'max':
            if monitor_value > self.best + self.min_delta:
                self.wait = 0
                self.best = monitor_value
            else:
                self.wait += 1

        if self.wait >= self.patience:
            print(f'Early stopping at epoch {epoch + 1}\n')
            self.model.stop_training = True

# 保存权重
save_callback = keras.callbacks.ModelCheckpoint(
    os.path.join(directory, checkpoint_dir, checkpoint_fn),
    monitor='accuracy',
    save_best_only=True,
    mode='max',
    verbose=VERBOSE_MODE
)

# tensorboard可视化
tensorboard_callback = keras.callbacks.TensorBoard(
    os.path.join(directory, tensorboard_dir),
    histogram_freq=1
)


# 模型类
class SkinCancerModel:
    def __init__(self,
                 learning_rate=LEARNING_RATE):
        self.learning_rate = learning_rate
        self.model = self._create_model()
    
    # 构建模型
    def _create_model(self):
        model = create_model()
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
            optimizer=keras.optimizer_v2.adam.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        return model
    
    def train(self,
              train_datas,
              train_labels,
              validation_datas,
              validation_labels,
              batch_size,
              validation_batch_size,
              epochs=1,
              class_weight=None,
              callbacks=None):
        if callbacks is None:
            callbacks = []
        self._train_in_batches(
            train_datas,
            train_labels,
            validation_datas,
            validation_labels,
            batch_size,
            validation_batch_size,
            epochs,
            class_weight,
            callbacks
        )

    # 手动分批次训练
    def _train_in_batches(self,
                          train_datas,
                          train_labels,
                          validation_datas,
                          validation_labels,
                          batch_size,
                          validation_batch_size,
                          epochs,
                          class_weight,
                          callbacks):
        num_batches = len(train_labels) // batch_size
        num_val_batches = len(validation_labels) // validation_batch_size

        if len(train_labels) % batch_size != 0: num_batches += 1
        if len(validation_labels) % validation_batch_size != 0: num_val_batches += 1

        # 回调类预处理
        for callback in callbacks:

            # 为每个回调类设定模型
            callback.set_model(self.model)

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            start = time.time()

            # 调用学习率调整器
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_begin'):
                    callback.on_epoch_begin(epoch)

            epoch_loss, epoch_accuracy = 0, 0
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, len(train_labels))

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
                batch_loss, batch_accuracy = self.model.train_on_batch(
                    batch_inputs,
                    batch_targets,
                    class_weight=class_weight
                )

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
                val_batch_end = min((i + 1) * validation_batch_size, len(validation_labels))

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
                    self.model.evaluate(val_inputs, val_targets, verbose=VERBOSE_MODE)
                
                val_loss += val_batch_loss
                val_accuracy += val_batch_accuracy

            # 计算验证集的loss和accuracy
            val_loss /= num_val_batches
            val_accuracy /= num_val_batches
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

            # 调用精度调整器
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(
                        epoch, 
                        logs={'accuracy': epoch_accuracy, 'val_accuracy': val_accuracy}
                    )
            
            if self.model.stop_training == True:
                print('\nStopping training early bdue to a callback\n')
                break

            # 一个epoch的耗时
            end = time.time()
            epoch_duration = end - start
            print(f'Epoch {epoch + 1} takes {epoch_duration:.2f} seconds\n')
    
    def predict(self,
                test_datas,
                test_batch_size):
        self._predict_in_batches(
            test_datas,
            test_batch_size
        )

    # 预测
    def _predict_in_batches(self,
                           test_datas,
                           test_batch_size):
        print('Start predicting\n')
        start = time.time()
        num_test_batches = len(test_datas[0]) // test_batch_size
        if len(test_datas[0]) % test_batch_size != 0: num_test_batches += 1

        predictions = []
        # 分批次进数据
        for i in range(num_test_batches):
            test_batch_start = i * test_batch_size
            test_batch_end = min((i + 1) * test_batch_size, len(test_datas[0]))

            test_batch_data = [data[test_batch_start: test_batch_end] for data in test_datas]

            test_data = []
            for image, age, site, sex in zip(*test_batch_data):
                processed_test_datas, _ = \
                    preprocess_test_data((image, age, site, sex), None)
                test_data.append(processed_test_datas)

            test_inputs = []
            for inputs in test_data:
                test_inputs.append(inputs['data_input'].numpy())
            
            test_inputs = np.array(test_inputs)

            batch_predictions = self.model.predict(test_inputs, verbose=VERBOSE_MODE)

            # 将预测的概率转化为实际的标签
            for sample_prediction in batch_predictions:
                prediction_label = label_names[np.argmax(sample_prediction)]
                predictions.append(prediction_label)
        for prediction_label in predictions:
            print(prediction_label)
        print('\n')

        end = time.time()
        prediction_duration = end - start
        print(f'Predicting takes {prediction_duration:.2f} seconds\n')

    # 保存模型
    def save_model(self, filepath):
        self.model.save(filepath)

    # 保存权重
    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    # 加载模型
    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)

    # 加载权重
    def load_weights(self, filepath):
        self.model = keras.Model.load_weights(filepath)