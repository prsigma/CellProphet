# -*- coding: utf-8 -*-      
import torch
import numpy as np
import time 
import os
from models.granger_model import GRANGER, train_phase1

def run_granger_on_dataset(dataset_name):
    """
    在指定的数据集上运行GRANGER模型
    """
    device = torch.device('cuda')
    
    # 构建数据路径（相对于项目根目录）
    data_path = f'/home/pengrui/TRIGON_rebuttal/BEELINE-data/{dataset_name}/time_output.npy'
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件 {data_path} 不存在")
        return False
    
    print(f"正在处理数据集: {dataset_name}")
    print(f"数据路径: {data_path}")
    
    try:
        X_np = np.load(data_path).T 
        print(f"数据形状: {X_np.shape}")
        
        dim = X_np.shape[-1] 
        GC = np.zeros([dim,dim])
        for i in range(dim):
            GC[i,i] = 1
            if i!=0:
                GC[i,i-1] = 1
        
        X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)
        smaple=X.shape[0]
        full_connect = np.ones(GC.shape)
        granger = GRANGER(X_np,X.shape[-1],full_connect, hidden=256).cuda(device=device)

        start_time = time.time() 
        train_loss_list = train_phase1(
            granger, X, context=20, lam=0.3, lam_ridge=0, lr=3e-3, max_iter=500, 
            check_every=50,batch_size=8)
        end_time = time.time() 
        print(f"数据集 {dataset_name} 处理完成")
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        
        # 保存结果到对应的文件夹
        output_dir = f'BEELINE-data/{dataset_name}'
        
        # 获取Granger因果关系矩阵
        gc_matrix = granger.GC(threshold=False).detach().cpu().numpy()
        gc_result_path = os.path.join(output_dir, 'granger_causality_matrix.csv')
        np.savetxt(gc_result_path, gc_matrix, delimiter=',', fmt='%.6f')
        print(f"Granger因果关系矩阵已保存到: {gc_result_path}")
        
        return True
        
    except Exception as e:
        print(f"处理数据集 {dataset_name} 时发生错误: {str(e)}")
        return False

def batch_process_all_datasets():
    """
    批量处理所有数据集
    """
    # 可用的数据集列表
    available_datasets = ['mDC', 'hHep', 'hESC', 'mESC', 'mHSC-L', 'mHSC-GM', 'mHSC-E']
    
    print("="*50)
    print("开始批量处理所有BEELINE数据集")
    print("="*50)
    
    success_count = 0
    total_count = len(available_datasets)
    
    for i, dataset in enumerate(available_datasets, 1):
        print(f"\n[{i}/{total_count}] 开始处理数据集: {dataset}")
        print("-" * 40)
        
        success = run_granger_on_dataset(dataset)
        if success:
            success_count += 1
            print(f"✅ 数据集 {dataset} 处理成功")
        else:
            print(f"❌ 数据集 {dataset} 处理失败")
        
        print("-" * 40)
    
    print(f"\n{'='*50}")
    print(f"批量处理完成!")
    print(f"成功处理: {success_count}/{total_count} 个数据集")
    print(f"{'='*50}")

if __name__ == "__main__":
    batch_process_all_datasets() 