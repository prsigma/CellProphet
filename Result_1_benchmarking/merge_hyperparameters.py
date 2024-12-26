import os
import pandas as pd

# 定义一个函数来处理每一行并将其转换为字典
def process_line(line):
    # 分割每一行的键值对
    items = line.strip().split(',')
    # 将键值对转换为字典并返回
    return dict(item.split(':') for item in items)

# datasets = ["mESC", "mDC", "mHSC-GM", "mHSC-L", "mHSC-E", "hHep", "hESC"]
datasets = ["mESC", "mHSC-GM", "mHSC-L", "mHSC-E", "hHep"]

# 初始化一个空的DataFrame，用于存储所有数据
all_data_auc_df = pd.DataFrame()
all_data_aupr_df = pd.DataFrame()

for each_data in datasets:
    file_path = os.path.join('/home/pengrui/STGRN/output', each_data, 'MTGRN', 'results_0.txt')
    
    # 读取文件内容并逐行处理
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # 存储处理后的数据
    processed_data = []

    for line in lines:
        if line.strip():  # 确保不处理空行
            processed_line = process_line(line)
            processed_data.append(processed_line)
    
    # 将处理后的数据转换为DataFrame
    df = pd.DataFrame(processed_data)
    
    # 将列名转换为适当的数据类型（如果需要）
    df['input_length'] = df['input_length'].astype(int)
    df['predict_length'] = df['predict_length'].astype(int)
    df['d_model'] = df['d_model'].astype(int)
    df['heads'] = df['heads'].astype(int)
    df['dropout'] = df['dropout'].astype(float)
    df['layers'] = df['layers'].astype(int)
    df['auc'] = df['auc'].astype(float)
    df['aupr'] = df['aupr'].astype(float)

    # 将这个数据集的DataFrame添加到总的DataFrame中
    all_data_auc_df = pd.concat([all_data_auc_df, df['auc']],axis=1)
    all_data_aupr_df = pd.concat([all_data_aupr_df, df['aupr']],axis=1)

all_data_auc_df.mean(axis=1)
all_data_aupr_df.mean(axis=1)