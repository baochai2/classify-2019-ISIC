import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.ERROR)

import pandas as pd
import numpy as np

from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sklearn.utils.class_weight import compute_class_weight

from utils_for_clsfy_skin_cancer import (
    directory, data_dir,
    train_data_fn, train_label_fn, train_input_dir,
    checkpoint_dir, tensorboard_dir,
    saved_model_dir,
    configure_GPU,
    ensure_dir,
    encode_age, encode_site, encode_sex,
    parse_label,
    reduce_lr, AccuracyCallback, EarlyStoppingCallback,
    save_callback, tensorboard_callback,
    SkinCancerModel
)



# 配置GPU
configure_GPU()


# 使用混合精度训练
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


# 超参数
TRAIN_VALIDATION_RATE = 0.9
BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 32
EPOCHS = 40


# 确保路径
ensure_dir(
    [
        os.path.join(directory, checkpoint_dir),
        os.path.join(directory, tensorboard_dir),
        os.path.join(directory, saved_model_dir)
    ]
)


# 读取数据
df_data = pd.read_csv(os.path.join(directory, data_dir, train_data_fn))
df_label = pd.read_csv(os.path.join(directory, data_dir, train_label_fn))


# 打乱数据
df_data = df_data.sample(frac=1, random_state=42).reset_index(drop=True)
df_label = df_label.sample(frac=1, random_state=42).reset_index(drop=True)


# 提取数据特征
train_image_fp = df_data['image']\
    .apply(lambda x: os.path.join(directory, data_dir, train_input_dir, x + '.jpg')).values  # 病变图像路径
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


# 计算标签权重
train_labels = np.array(train_label)
computed_weights = compute_class_weight('balanced', classes=np.unique(train_label), y=train_label)
class_weight = {i: computed_weights[i] for i in range(len(computed_weights))}


# 初始化模型
model = SkinCancerModel()


# 训练数据
model.train(
    [train_image_fp, train_age, train_site, train_sex],
    train_label,
    [validation_image_fp, validation_age, validation_site, validation_sex],
    validation_label,
    batch_size=BATCH_SIZE,
    validation_batch_size=VALIDATION_BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=
    [
        reduce_lr,
        AccuracyCallback(),
        EarlyStoppingCallback(monitor='val_accuracy', mode='max', patience=None),
        save_callback,
        tensorboard_callback
    ]
)


# 保存模型
model.save_model(os.path.join(directory, saved_model_dir, 'completed_model.keras'))