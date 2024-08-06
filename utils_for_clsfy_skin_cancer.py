import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pandas as pd
import numpy as np
import keras

from keras import layers


# 超参数
HEIGHT = 256
WIDTH = 256


# 数据路径
dir = 'skin_cancer/'

train_data_fn = 'ISIC_2019_Training_Metadata.csv'
train_label_fn = 'ISIC_2019_Training_GroundTruth.csv'
train_input_dir = 'ISIC_2019_Training_Input'

test_data_fn = 'ISIC_2019_Test_Metadata.csv'
test_input_dir = 'ISIC_2019_Test_Input'

train_cache_dir = 'E:/code_cache/train_cache/'
validation_cache_dir = 'E:/code_cache/validation_cache/'
test_cache_dir = 'E:/code_cache/test_cache/'


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


# 编码未处理的数据特征
def encode_age(dataframe):
    age = np.array([], dtype='int32')  # 初始化存储特征的列表
    for row in np.array(dataframe['age_approx']):
        if pd.isna(row): row = 0  # 如果row为缺失值NaN
        age = np.append(age, int(row))
    return age

def encode_site(dataframe):
    site = np.array([], dtype='int32')
    for row in np.array(dataframe['anatom_site_general']):
        idx = 0 if pd.isna(row) else site_names.index(row)
        site = np.append(site, idx)
    return site

def encode_sex(dataframe):
    sex = np.array([], dtype='int32')
    for row in np.array(dataframe['sex']):
        if pd.isna(row):
            row = 0
        elif row == 'male':
            row = 1
        else:
            row = 2
        sex = np.append(sex, row)
    return sex


# 解析出文件的标签
def parse_label(dataframe):
    label = np.array([], dtype='int64')
    for row in np.array(dataframe[label_names]):  # 按行遍历每一行的病种
        idx = np.where(row == 1.0)[0][0]  # 找到值为1的列索引
        label = np.append(label, idx)
    return label


# 预处理
# 将读取到的图像名转化为张量
def read_image(image, age, site, sex, label):
    image = tf.io.read_file(image)  # 读取文件

    image = tf.image.decode_jpeg(image, channels=3)
        # image数据是由height*width个channel张量组成
    return image, age, site, sex, label

# 标准化
def normalize_image(image, age, site, sex, label):
    image = tf.image.resize(image, (HEIGHT, WIDTH))  # 可调整插值方法method
    return tf.cast(image, dtype=tf.float32) / 255.0, age, site, sex, label

# 数据增强
def augment(image, age, site, sex, label):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_saturation(image, 5, 10)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, age, site, sex, label

# 将特征合并为输入数据
def build_inputs(image, age, site, sex, label):
    inputs = {
        'image_input': image,
        'age_input': tf.expand_dims(age, -1),
        'site_input': tf.expand_dims(site, -1),
        'sex_input': tf.expand_dims(sex, -1)
    }  # 传入字典的键应该与模型输入层的name相同
    return inputs, label

def preprocess_train_data(image, age, site, sex, label):
    image, age, site, sex, label  = read_image(image, age, site, sex, label)
    image, age, site, sex, label = normalize_image(image, age, site, sex, label)
    image, age, site, sex, label = augment(image, age, site, sex, label)
    inputs, label = build_inputs(image, age, site, sex, label)
    return inputs, label

def preprocess_validation_data(image, age, site, sex, label):
    image, age, site, sex, label  = read_image(image, age, site, sex, label)
    image, age, site, sex, label = normalize_image(image, age, site, sex, label)
    inputs, label = build_inputs(image, age, site, sex, label)
    return inputs, label


# 构建模型
def create_model():

    # 图像输入
    image_inputs = keras.Input(shape=(HEIGHT, WIDTH, 3), name='image_input')
    x = layers.Conv2D(32, 3, activation='relu')(image_inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)

    # 其他特征输入
    age_inputs = keras.Input(shape=(1,), name='age_input')
    site_inputs = keras.Input(shape=(1,), name='site_input')
    sex_inputs = keras.Input(shape=(1,), name='sex_input')

    # 合并特征
    concatenation = layers.Concatenate(axis=1)([x, age_inputs, site_inputs, sex_inputs])

    x = layers.Dense(128, activation='relu')(concatenation)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(len(label_names), activation='softmax')(x)
    model = keras.Model(
        inputs={
        'image_input': image_inputs,
        'age_input': age_inputs,
        'site_input': site_inputs,
        'sex_input': sex_inputs
        },
        outputs=outputs
    )
    return model