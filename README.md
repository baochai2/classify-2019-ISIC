# Project Introduction

The project aims to **classify** types of **skin cancer** and **predict** which cancer patients have.

## 1. Dataset

### 1.1 [Click this to download dataset](https://challenge.isic-archive.com/data/#2019)

### 1.2 How to arrange dataset?

   After downloading the dataset, follow these steps to arrange it properly:
   
   1. Unzip the downloaded file.
   2. Organize the files into the following structure:
      ```
      skin_cancer/
      ├── ISIC_2019_Training_Input/
      │   ├── ISIC_0000000.jpg
      │   ├── ISIC_0000001.jpg
      │   └── ISIC_0000002.jpg
      ├── ISIC_2019_Test_Input/
      │   ├── ISIC_0034321.jpg
      │   ├── ISIC_0034322.jpg
      │   └── ISIC_0034323.jpg
      ├── ISIC_2019_Training_Metadata.csv
      ├── ISIC_2019_Training_GroundTruth.csv
      └── ISIC_2019_Test_Metadata.csv
      ```


## 2. Label Interpretion

### 2.1 ISIC_2019_Training_Metadata.csv

-  **image**: 图像的唯一标识符

-  **age_approx**: 病人的大致年龄，单位为岁

-  **anatom_site_general**: 病变部位的描述

-  **lesion_id**: 病变数据的来源

-  **sex**: 病人的性别

#### 2.1.1 anatom_site_general

-  **NaN**：缺失值，表示没有记录具体的解剖部位信息

-  **oral/genital**：口腔或生殖器部位

-  **lower extremity**：下肢，包括大腿、小腿和脚

-  **palms/soles**：手掌或脚掌

-  **lateral torso**：身体侧面，包括腰部、侧胸部等区域

-  **posterior torso**：身体背部，包括后背、肩胛骨区域等

-  **head/neck**：头部和颈部

-  **anterior torso**：身体前部，包括胸部和腹部

-  **upper extremity**：上肢，包括肩膀、手臂、前臂和手

### 2.2 ISIC_2019_Training_GroundTruth.csv

-  **MEL**: Melanoma（黑色素瘤）

-  **NV**: Melanocytic Nevi（黑色素细胞痣）

-  **BCC**: Basal Cell Carcinoma（基底细胞癌）

-  **AK**: Actinic Keratosis（光化性角化病）

-  **BKL**: Benign Keratosis-like Lesions（良性角化病样病变）

-  **DF**: Dermatofibroma（皮肤纤维瘤）

-  **VASC**: Vascular Lesions（血管性病变）

-  **SCC**: Squamous Cell Carcinoma（鳞状细胞癌）

-  **UNK**: Unknown（未知）
