import pandas as pd
import numpy as np
import os
from models.utils import get_origin_expression_data

def time_rank(expression_path, PseudoTime_path, output_dir):
    """
    对表达数据按照伪时间排序
    """
    expression_df = pd.read_csv(expression_path, index_col=0)
    pseudotime_df = pd.read_csv(PseudoTime_path, index_col=0)
    
    # 检查列名并设置valid_time
    if 'PseudoTime' in pseudotime_df.columns:
        pseudotime_df['valid_time'] = pseudotime_df['PseudoTime']
    elif 'PseudoTime1' in pseudotime_df.columns:
        pseudotime_df['valid_time'] = pseudotime_df['PseudoTime1'].fillna(pseudotime_df['PseudoTime2']) if 'PseudoTime2' in pseudotime_df.columns else pseudotime_df['PseudoTime1']
    else:
        # 如果没有找到预期的列名，使用第一列
        pseudotime_df['valid_time'] = pseudotime_df.iloc[:, 0]
    
    sorted_pseudotime_df = pseudotime_df.sort_values(by='valid_time')
    sorted_cellids = sorted_pseudotime_df.index.tolist()
    sorted_expression_df = expression_df[sorted_cellids]
    
    # 保存排序后的时间序列数据
    time_csv_path = os.path.join(output_dir, 'time.csv')
    sorted_expression_df.to_csv(time_csv_path)
    return sorted_expression_df, time_csv_path

def to_npy(expression_path, output_dir):
    """
    将CSV文件转换为NPY格式
    """
    a, b = get_origin_expression_data(expression_path)
    arr = np.vstack(list(a.values()))
    npy_path = os.path.join(output_dir, 'time_output.npy')
    np.save(npy_path, arr)
    return npy_path

def process_single_folder(folder_path):
    """
    处理单个文件夹
    """
    print(f"正在处理文件夹: {folder_path}")
    
    # 构建文件路径
    expression_path = os.path.join(folder_path, 'ExpressionData.csv')
    pseudotime_path = os.path.join(folder_path, 'PseudoTime.csv')
    
    # 检查文件是否存在
    if not os.path.exists(expression_path):
        print(f"警告: {expression_path} 不存在，跳过此文件夹")
        return False
    if not os.path.exists(pseudotime_path):
        print(f"警告: {pseudotime_path} 不存在，跳过此文件夹")
        return False
    
    try:
        # 执行时间排序
        sorted_expression_df, time_csv_path = time_rank(expression_path, pseudotime_path, folder_path)
        print(f"  - 生成时间排序文件: {time_csv_path}")
        
        # 转换为NPY格式
        npy_path = to_npy(time_csv_path, folder_path)
        print(f"  - 生成NPY文件: {npy_path}")
        
        return True
        
    except Exception as e:
        print(f"错误: 处理文件夹 {folder_path} 时出现异常: {str(e)}")
        return False

def process_all_folders(base_dir='BEELINE-data'):
    """
    批量处理所有文件夹
    """
    if not os.path.exists(base_dir):
        print(f"错误: 基础目录 {base_dir} 不存在")
        return
    
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    if not subfolders:
        print(f"错误: 在 {base_dir} 中没有找到子文件夹")
        return
    
    print(f"找到 {len(subfolders)} 个子文件夹: {subfolders}")
    
    success_count = 0
    total_count = len(subfolders)
    
    # 处理每个子文件夹
    for subfolder in subfolders:
        folder_path = os.path.join(base_dir, subfolder)
        success = process_single_folder(folder_path)
        if success:
            success_count += 1
        print()  # 添加空行分隔
    
    print(f"处理完成! 成功处理 {success_count}/{total_count} 个文件夹")

if __name__ == "__main__":
    # 批量处理所有文件夹
    process_all_folders()