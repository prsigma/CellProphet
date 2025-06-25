# -*- coding: utf-8 -*-      
import torch
import numpy as np
import pandas as pd
import time 
import os
from models.granger_model import GRANGER, train_phase1
from tqdm import tqdm

def run_granger_on_dataset(dataset_name):
    """
    在指定的数据集上运行GRANGER模型
    """
    device = torch.device('cuda:0')
    
    # 构建数据路径（相对于项目根目录）
    data_path = f'/home/pengrui/TRIGON_rebuttal/output/{dataset_name}/TRIGON/trigon_expression.csv'
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件 {data_path} 不存在")
        return None, None
    
    print(f"正在处理数据集: {dataset_name}")
    print(f"数据路径: {data_path}")
    
    # 从CSV文件加载数据
    try:
        print("正在加载CSV文件...")
        # 读取CSV文件，第一行作为header，第一列作为index
        df = pd.read_csv(data_path, index_col=0)
        
        # # 随机选择1500列
        # if df.shape[1] > 1500:
        #     print(f"原始数据有 {df.shape[1]} 列，随机选择1500列")
        #     np.random.seed(42)  # 设置随机种子以确保结果可重现
        #     selected_cols = np.random.choice(df.columns, size=1500, replace=False)
        #     df = df[selected_cols]
        #     print(f"选择后数据形状: {df.shape}")
        # else:
        #     print(f"数据只有 {df.shape[1]} 列，保持原样")
        
        print(f"CSV文件形状: {df.shape}")
        print(f"数据类型: {df.dtypes.unique()}")
        
        # 保存基因名称用于后续结果保存
        gene_names = df.columns.tolist()

        # 转换为numpy数组
        X_np = df.values
        print(f"转换为numpy后形状: {X_np.shape}")
        print(f"numpy数据类型: {X_np.dtype}")
        
        # 释放dataframe内存
        del df
        
        # 处理非数值数据
        print("开始清理数据...")
        
        # 如果数据类型是object，需要逐个处理
        if X_np.dtype == 'object' or not np.issubdtype(X_np.dtype, np.number):
            print("检测到非数值类型数据，开始转换...")
            
            # 创建一个新的数值数组
            X_clean = np.zeros(X_np.shape, dtype=np.float32)
            
            for i in range(X_np.shape[0]):
                for j in range(X_np.shape[1]):
                    try:
                        # 尝试转换每个元素
                        val = X_np[i, j]
                        if pd.isna(val) or val == '' or val == 'nan' or val == 'NaN':
                            X_clean[i, j] = 0.0
                        else:
                            X_clean[i, j] = float(val)
                    except (ValueError, TypeError):
                        # 如果无法转换，设为0
                        X_clean[i, j] = 0.0
                        
                # 显示进度
                if i % 100 == 0:
                    print(f"数据清理进度: {i+1}/{X_np.shape[0]} 行")
            
            X_np = X_clean
            print("数据清理完成")
        else:
            # 如果已经是数值类型，直接转换
            X_np = X_np.astype(np.float32)
        
        print(f"清理后数据形状: {X_np.shape}, 数据类型: {X_np.dtype}")
        
        # 检查是否有无穷大或NaN值
        inf_count = np.sum(np.isinf(X_np))
        nan_count = np.sum(np.isnan(X_np))
        
        if inf_count > 0 or nan_count > 0:
            print(f"警告: 发现 {inf_count} 个无穷大值和 {nan_count} 个NaN值，进行清理...")
            X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6)
            print("数值清理完成")
        
        # 确保数据是float32类型
        X_np = X_np.astype(np.float32)
        print(f"最终数据形状: {X_np.shape}, 数据类型: {X_np.dtype}")
        
        # 检查数据范围和统计信息
        print(f"数据范围: [{np.min(X_np):.6f}, {np.max(X_np):.6f}]")
        print(f"数据均值: {np.mean(X_np):.6f}, 标准差: {np.std(X_np):.6f}")
        print(f"零值比例: {np.mean(X_np == 0):.4f}")
        
        # 验证数据的有效性
        if X_np.shape[0] == 0 or X_np.shape[1] == 0:
            print("错误: 数据为空")
            return None, None
            
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    dim = X_np.shape[-1] 
    GC = np.zeros([dim,dim])
    for i in range(dim):
        GC[i,i] = 1
        if i!=0:
            GC[i,i-1] = 1
    
    # 安全地转换为torch tensor
    try:
        print("正在创建torch tensor...")
        # 直接在指定设备上创建tensor
        X_cpu = torch.tensor(X_np[np.newaxis], dtype=torch.float32)
        print(f"CPU tensor创建成功，形状: {X_cpu.shape}")
        
        # 移动到指定设备
        X = X_cpu.to(device)
        print(f"设备tensor创建成功，形状: {X.shape}, 设备: {X.device}")
        
        # 释放CPU tensor
        del X_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 验证tensor的有效性
        if torch.any(torch.isnan(X)) or torch.any(torch.isinf(X)):
            print("警告: tensor包含NaN或Inf值")
            X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            print("tensor清理完成")
            
    except Exception as e:
        print(f"创建torch tensor时发生错误: {e}")
        print(f"尝试的数据形状: {X_np[np.newaxis].shape}")
        print(f"数据类型: {X_np.dtype}")
        print(f"数据示例: {X_np[:2, :5]}")  # 显示前2行5列作为示例
        import traceback
        traceback.print_exc()
        return None, None
    
    smaple=X.shape[0]
    full_connect = np.ones(GC.shape)
    
    print(f"初始化GRANGER模型...")
    try:
        # 传递device参数给GRANGER模型
        granger = GRANGER(X_np, X.shape[-1], full_connect, hidden=256, device=device).to(device)
        print("GRANGER模型初始化成功")
        
        # 检查内存使用
        if torch.cuda.is_available():
            print(f"模型初始化后内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
    except Exception as e:
        print(f"初始化GRANGER模型时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    print(f"开始训练GRANGER模型...")
    
    # 根据数据大小调整训练参数
    batch_size = 2
    context = 15
    max_iter = 100
    print(f"大数据集，使用较小参数: batch_size={batch_size}, context={context}, max_iter={max_iter}")

    print(f"训练参数: context={context}, lam=0.3, lr=3e-3, max_iter={max_iter}, batch_size={batch_size}")
    
    start_time = time.time() 
    try:
        train_loss_list = train_phase1(
            granger, X, context=context, lam=0.3, lam_ridge=0, lr=3e-3, max_iter=max_iter, 
            check_every=50, batch_size=batch_size)
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理内存后重试一次
        print("清理内存后重试...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            train_loss_list = train_phase1(
                granger, X, context=max(10, context-5), lam=0.3, lam_ridge=0, lr=3e-3, max_iter=max_iter//2, 
                check_every=50, batch_size=max(2, batch_size//2))
        except Exception as e2:
            print(f"重试失败: {e2}")
            return None, None
        
    end_time = time.time() 
    print(f"数据集 {dataset_name} 处理完成")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    
    # 保存结果到对应的文件夹
    output_dir = f'/home/pengrui/TRIGON_rebuttal/output/{dataset_name}/GRANGER'
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取Granger因果关系矩阵
    try:
        gc_matrix = granger.GC(threshold=False).detach().cpu().numpy()
        
        # 保存原始矩阵格式
        gc_result_path = os.path.join(output_dir, 'granger_causality_matrix.csv')
        np.savetxt(gc_result_path, gc_matrix, delimiter=',', fmt='%.6f')
        print(f"Granger因果关系矩阵已保存到: {gc_result_path}")
        
        # 转换为三列格式并保存
        regulatory_pairs = []
        for i in range(gc_matrix.shape[0]):  # 调控基因
            for j in range(gc_matrix.shape[1]):  # 靶基因
                if i != j:  # 排除自调控
                    regulatory_gene = gene_names[i]
                    target_gene = gene_names[j]
                    regulation_value = gc_matrix[i, j]
                    regulatory_pairs.append([regulatory_gene, target_gene, regulation_value])
        
        # 创建DataFrame并保存
        regulatory_df = pd.DataFrame(regulatory_pairs, columns=['Gene1', 'Gene2', 'weighted_value'])
        three_col_path = os.path.join(output_dir, 'grn.csv')
        regulatory_df.to_csv(three_col_path, index=False)
        print(f"三列格式的调控关系已保存到: {three_col_path}")
        
    except Exception as e:
        print(f"保存结果时发生错误: {e}")
    
    # 保存返回值
    result_granger = granger
    result_loss = train_loss_list if 'train_loss_list' in locals() else []
    
    # 清理内存
    del granger, X
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result_granger, result_loss

if __name__ == "__main__":
    # 可用的数据集列表
    # available_datasets = ['mESC','mDC', 'hHep', 'hESC','mHSC-L', 'mHSC-GM']
    # available_datasets = ['mHSC-E']
    available_datasets = ['hESC']
    
    print(f"可用数据集: {available_datasets}")
    print(f"总共需要处理 {len(available_datasets)} 个数据集")
    print(f"开始循环处理所有数据集...")
    
    # 创建总体进度条
    dataset_pbar = tqdm(available_datasets, desc="数据集进度", position=0)
    
    # 循环处理所有数据集
    for dataset_idx, dataset_name in enumerate(dataset_pbar):
        dataset_pbar.set_description(f"处理数据集 {dataset_idx+1}/{len(available_datasets)}: {dataset_name}")
        
        print(f"\n{'='*50}")
        print(f"正在处理数据集: {dataset_name} ({dataset_idx+1}/{len(available_datasets)})")
        print(f"{'='*50}")
        
        try:
            # 运行GRANGER模型
            granger_model, loss_list = run_granger_on_dataset(dataset_name)
            
            if granger_model is not None:
                print(f"数据集 {dataset_name} 的GRANGER模型训练完成！")
            else:
                print(f"数据集 {dataset_name} 的GRANGER模型训练失败！")
                
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时发生错误: {str(e)}")
            continue
    
    dataset_pbar.close()
    print(f"\n{'='*50}")
    print("所有数据集处理完成！")
    print(f"{'='*50}")