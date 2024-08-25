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
from keras.metrics import TopKCategoricalAccuracy
from keras.utils.all_utils import to_categorical



# 超参数
HEIGHT = 224
WIDTH = 224
CHANNEL = 3
SMOOTHING_FACTOR = 0.1
BETA_PARAMETER = 0.2
WARMUP_STEPS = 2000
NUM_K = 2  # top-k准确率的k
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

output_dir = 'output/'
log_file = os.path.join(directory, output_dir, 'train results.txt')


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

    # 旋转图像
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # 调整亮度 饱和度 色调
    brightness_factor = tf.random.uniform([], 0.6, 1.4)
    image = tf.image.adjust_brightness(image, brightness_factor - 1.0)

    saturation_factor = tf.random.uniform([], 0.6, 1.4)
    image = tf.image.adjust_saturation(image, saturation_factor)

    hue_factor = tf.random.uniform([], -0.4, 0.4)
    image = tf.image.adjust_hue(image, hue_factor)

    # 加入高斯噪声
    noise = tf.random.normal(shape=tf.shape(image), mean=0, stddev=0.1, dtype=tf.float64)
    image = tf.add(image, noise)
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

# 标签平滑化
def label_smoothing(label, smoothing_factor=SMOOTHING_FACTOR):
    if label:
        label = (1 - smoothing_factor) * label + smoothing_factor / len(label_names)
    return label

def preprocess_data(data, label, is_training=True):
    image, age, site, sex = data
    image  = read_image(image)
    image = normalize_image(image)
    if is_training:
        image = augment(image)
    data = merge(image, age, site, sex)
    ensure_data(data)
    inputs = build_inputs(data)
    label = label_smoothing(label)
    return inputs, label

# 混合数据
def mixup(x1, y1, x2, y2, alpha=BETA_PARAMETER):
    lambda_parameter = np.random.beta(alpha, alpha)
    mixed_x = lambda_parameter * x1 + (1 - lambda_parameter) * x2
    mixed_y = lambda_parameter * y1 + (1 - lambda_parameter) * y2
    return mixed_x, mixed_y

def preprocess_train_data(data, label, train_datas, train_labels):

    # 抽取混合的第二组数据
    random_idx = np.random.randint(0, len(train_datas))
    image2 = train_datas[0][random_idx]
    age2 = train_datas[1][random_idx]
    site2 = train_datas[2][random_idx]
    sex2 = train_datas[3][random_idx]
    label2 = train_labels[random_idx]

    # 处理数据
    input1, label1 = preprocess_data(data, label)
    input2, label2 = preprocess_data((image2, age2, site2, sex2), label2)

    # 混合数据
    inputs, label = mixup(input1['data_input'], label1, input2['data_input'], label2)

    ensure_data(inputs)
    inputs = build_inputs(inputs)
    return inputs, label

# 直接进入模型的数据集接口
def process_dataset(batch, batch_size, datas, labels, is_training=True):
    batch_start = batch * batch_size
    batch_end = min((batch + 1) * batch_size, len(labels))

    # 获取当前批次的数据
    batch_data = tuple([data[batch_start: batch_end] for data in datas])
    batch_label = labels[batch_start: batch_end]

    # 预处理
    batch_inputs, batch_targets = [], []
    if is_training:
        for image, age, site, sex, label in zip(*batch_data, batch_label):
            processed_batch_datas, processed_batch_labels = \
                preprocess_train_data((image, age, site, sex), label, datas, labels)
            batch_inputs.append(processed_batch_datas['data_input'].numpy())
            batch_targets.append(processed_batch_labels)
    else:
        for image, age, site, sex, label in zip(*batch_data, batch_label):
            processed_batch_datas, processed_batch_labels = \
                preprocess_data((image, age, site, sex), label, is_training=is_training)
            batch_inputs.append(processed_batch_datas['data_input'].numpy())
            batch_targets.append(processed_batch_labels)

    batch_inputs = np.array(batch_inputs)
    batch_targets = np.array(batch_targets)
    batch_targets = to_categorical(batch_targets, num_classes=len(label_names))
    return batch_inputs, batch_targets


# 写入日志
def logging_to_txt(
    epoch,
    epochs,
    epoch_loss,
    epoch_accuracy,
    epoch_top_k_accuracy,
    val_loss,
    val_accuracy,
    val_top_k_accuracy,
    epoch_duration
):
    with open(log_file, 'a') as f:
        f.write(f'\nEpoch {epoch + 1}/{epochs}\n')
        f.write(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, '
                f'Train top-k Accuracy: {epoch_top_k_accuracy:.4f}\n')
        f.write(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, '
                f'Validation top-k Accuracy: {val_top_k_accuracy:.4f}\n')
        f.write(f'Epoch {epoch + 1} takes {epoch_duration:.2f} seconds\n')


# 构建模型
def create_model():

    # 输入
    data_inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNEL + 1), name='data_input')
    x = layers.Conv2D(
        NUM_KERNELS // 8, 7, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(data_inputs)
    x = layers.BatchNormalization(gamma_initializer=keras.initializers.initializers_v2.Zeros())(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS // 4, 7, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization(gamma_initializer=keras.initializers.initializers_v2.Zeros())(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS // 4, 7, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization(gamma_initializer=keras.initializers.initializers_v2.Zeros())(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS // 2, 5, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization(gamma_initializer=keras.initializers.initializers_v2.Zeros())(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS // 2, 5, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization(gamma_initializer=keras.initializers.initializers_v2.Zeros())(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS, 5, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization(gamma_initializer=keras.initializers.initializers_v2.Zeros())(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(
        NUM_KERNELS, 3, padding='same', kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization(gamma_initializer=keras.initializers.initializers_v2.Zeros())(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        NUM_DENSE_UNITS, kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization(gamma_initializer=keras.initializers.initializers_v2.Zeros())(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(
        NUM_DENSE_UNITS // 2, kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization(gamma_initializer=keras.initializers.initializers_v2.Zeros())(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(
        NUM_DENSE_UNITS // 4, kernel_regularizer=regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization(gamma_initializer=keras.initializers.initializers_v2.Zeros())(x)
    x = layers.Activation('relu')(x)

    # 输出
    outputs = layers.Dense(len(label_names), activation='softmax')(x)
    model = keras.Model(inputs={'data_input': data_inputs}, outputs=outputs)
    return model


# 回调
# 调整学习率
# def scheduler(epoch, lr):
#     if epoch % 10 == 0:
#         return lr / (10 ** (lr // 10))  # 每10个epoch学习率除10
#     else:
#         return lr
# lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)

# 学习率预热
# class WarmupLearningRateScheduler(keras.callbacks.Callback):
#     def __init__(self,
#                  warmup_steps=WARMUP_STEPS,
#                  base_lr=1e-5,
#                  target_lr=LEARNING_RATE):
#         super(WarmupLearningRateScheduler).__init__()
#         self.warmup_steps = warmup_steps
#         self.base_lr = base_lr
#         self.target_lr = target_lr
    
#     def on_batch_begin(self, batch, logs=None):
#         if batch < self.warmup_steps:
#             lr = self.base_lr + (self.target_lr - self.base_lr) * (batch / self.warmup_steps)
#         else:
#             lr = self.target_lr

#         # 设置学习率
#         keras.backend.set_value(self.model.optimizer.lr, lr)

# 自动调整学习率
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=4,
    min_lr=1e-6
)

# 精度控制
class AccuracyCallback(keras.callbacks.Callback):
     def on_epoch_end(self, epoch, logs=None):
          if logs.get('accuracy') > 0.9:
               self.model.stop_training = True

# 早停
class EarlyStoppingCallback:
    def __init__(self, monitor='val_loss', mode='min', min_delta=0.0001, patience=None):
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
        
        if self.patience is None:
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
            loss=keras.losses.CategoricalCrossentropy(),
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

        # 创建或清空日志
        with open(log_file, 'w') as f:
            f.write('')

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            start = time.time()

            # 调用学习率调整器
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_begin'):
                    callback.on_epoch_begin(epoch)

            epoch_loss, epoch_accuracy = 0, 0
            epoch_top_k_acc = TopKCategoricalAccuracy(k=NUM_K)
            for i in range(num_batches):
                batch_inputs, batch_targets = process_dataset(
                    i,
                    batch_size,
                    train_datas,
                    train_labels
                )

                # 回调
                for callback in callbacks:
                    if hasattr(callback, 'on_batch_begin'):
                        callback.on_batch_begin(i)

                # 训练数据
                batch_loss, batch_accuracy = self.model.train_on_batch(
                    batch_inputs,
                    batch_targets,
                    class_weight=class_weight
                )

                # 计算top-k准确率
                batch_predictions = self.model.predict(batch_inputs)
                epoch_top_k_acc.update_state(batch_targets, batch_predictions)

                # 累加loss和accuracy
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy

            # 计算每个epoch的loss和accuracy
            epoch_loss /= num_batches
            epoch_accuracy /= num_batches
            epoch_top_k_accuracy = epoch_top_k_acc.result().numpy()
            print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, '
                  f'Train top-k Accuracy: {epoch_top_k_accuracy:.4f}')

            # 将验证集转为numpy数组
            val_loss, val_accuracy = 0, 0
            val_top_k_acc = keras.metrics.TopKCategoricalAccuracy(k=NUM_K)
            for i in range(num_val_batches):
                val_inputs, val_targets = process_dataset(
                    i,
                    validation_batch_size,
                    validation_datas,
                    validation_labels,
                    is_training=False
                )

                # 评估模型
                val_batch_loss, val_batch_accuracy= \
                    self.model.evaluate(val_inputs, val_targets, verbose=VERBOSE_MODE)
                
                # 计算top-k准确率
                val_predictions = self.model.predict(val_inputs)
                val_top_k_acc.update_state(val_targets, val_predictions)
                
                val_loss += val_batch_loss
                val_accuracy += val_batch_accuracy

            # 计算验证集的loss和accuracy
            val_loss /= num_val_batches
            val_accuracy /= num_val_batches
            val_top_k_accuracy = val_top_k_acc.result().numpy()
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, '
                  f'Validation top-k Accuracy: {val_top_k_accuracy:.4f}')

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

            # 训练信息写入txt文件
            logging_to_txt(
                epoch,
                epochs,
                epoch_loss,
                epoch_accuracy,
                epoch_top_k_accuracy,
                val_loss,
                val_accuracy,
                val_top_k_accuracy,
                epoch_duration
            )

    def predict(self,
                test_datas,
                test_batch_size):
        return self._predict_in_batches(
            test_datas,
            test_batch_size
        )

    # 预测标签值
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

            # 获取当前批次的数据
            test_batch_data = [data[test_batch_start: test_batch_end] for data in test_datas]

            # 预处理
            test_inputs = []
            for image, age, site, sex in zip(*test_batch_data):
                processed_test_datas, _ = \
                    preprocess_data((image, age, site, sex), None, is_training=False)
                test_inputs.append(processed_test_datas['data_input'].numpy())

            test_inputs = np.array(test_inputs)

            # 预测
            batch_predictions = self.model.predict(test_inputs, verbose=VERBOSE_MODE)

            # 将预测的概率转化为实际的标签
            for sample_prediction in batch_predictions:
                prediction_label = [label_names[top] \
                                    for top in np.argsort(sample_prediction)[-1: -(NUM_K + 1): -1]]
                predictions.append(prediction_label)
        for i, prediction_label in enumerate(predictions):
            print(f'Sample {i + 1} is predicted as {prediction_label}')
            if i >= 9:
                break
        print('\n')

        end = time.time()
        prediction_duration = end - start
        print(f'Predicting takes {prediction_duration:.2f} seconds\n')

        return predictions

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