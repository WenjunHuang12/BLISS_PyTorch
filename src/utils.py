import torch
import time
import numpy as np
import os, sys
import pdb
import math
# import struct
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.utils import murmurhash3_32 as mmh3


def savememmap(path, ar):
    if path[-4:] != '.dat':
        path = path + '.dat'
    shape = ar.shape
    dtype = ar.dtype
    fp = np.memmap(path, dtype=dtype, mode='w+', shape=(shape))
    fp[:] = ar[:]
    fp.flush()

def getTrueNNS(x_train, metric, K):
    os.environ['CUDA_VISIBLE_DEVICES'] = 'cpu'
    # device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    begin_time = time.time()
    batch_size = 1000
    # batch_size = 400
    N = x_train.shape[0]
    output = np.zeros([N, K], dtype=np.int32)  # For up to 2B entries

    # x_train_tensor = torch.from_numpy(x_train).float()
    x_train_tensor = torch.from_numpy(x_train).float()
    if metric == 'IP':
        W = x_train_tensor.T  # Shape: [feature_dim, num_samples]
        for i in range(N // batch_size):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_train_tensor[start_idx:end_idx]  # Shape: [batch_size, feature_dim]
            sim = torch.matmul(x_batch, W)  # Shape: [batch_size, num_samples]
            topk_indices = torch.topk(sim, K, dim=1, largest=True, sorted=False)[1]  # Indices of top K similarities
            output[start_idx:end_idx] = topk_indices.cpu().numpy()

    elif metric == 'L2':
        W = x_train_tensor.T  # Shape: [feature_dim, num_samples]
        W_norm = torch.sum(W * W, dim=0)  # Shape: [num_samples]
        for i in range(N // batch_size):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_train_tensor[start_idx:end_idx]  # Shape: [batch_size, feature_dim]
            sim = 2 * torch.matmul(x_batch, W) - W_norm  # Shape: [batch_size, num_samples]
            topk_indices = torch.topk(sim, K, dim=1, largest=True, sorted=False)[1]
            output[start_idx:end_idx] = topk_indices.cpu().numpy()

    elif metric == 'cosine':
        x_train_norm = x_train_tensor / x_train_tensor.norm(dim=1, keepdim=True)
        W = x_train_norm.T  # Shape: [feature_dim, num_samples]
        for i in range(N // batch_size):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = x_train_norm[start_idx:end_idx]  # Shape: [batch_size, feature_dim]
            sim = torch.matmul(x_batch, W)  # Shape: [batch_size, num_samples]
            topk_indices = torch.topk(sim, K, dim=1, largest=True, sorted=False)[1]
            output[start_idx:end_idx] = topk_indices.cpu().numpy()

    print(time.time() - begin_time)
    return output


# def create_universal_lookups(r, B, n_classes, lookups_loc):
#     c_o = lookups_loc + 'class_order_' + str(r) + '.npy'
#     ct = lookups_loc + 'counts_' + str(r) + '.npy'
#     b_o = lookups_loc + 'bucket_order_' + str(r) + '.npy'
#     if os.path.exists(c_o) and os.path.exists(ct) and os.path.exists(b_o):
#         print('init lookups exists')
#     else:
#         counts = np.zeros(B + 1, dtype=int)
#         bucket_order = np.zeros(n_classes, dtype=int)
#         for i in range(n_classes):
#             bucket = mmh3(i, seed=r) % B
#             bucket_order[i] = bucket
#             counts[bucket + 1] += 1
#         counts = np.cumsum(counts)
#         rolling_counts = np.zeros(B, dtype=int)
#         class_order = np.zeros(n_classes, dtype=int)
#         for i in range(n_classes):
#             temp = bucket_order[i]
#             class_order[counts[temp] + rolling_counts[temp]] = i
#             rolling_counts[temp] += 1
        
#         np.save(c_o, class_order)
#         np.save(ct, counts)
#         np.save(b_o, bucket_order)

def create_universal_lookups(r, B, n_classes, lookups_loc):
    """
    创建通用的查找表并保存为 .npy 文件。

    参数:
    - r (int): 种子值，用于哈希函数。
    - B (int): 桶的数量。
    - n_classes (int): 类别的数量。
    - lookups_loc (str): 查找表保存的目录路径。
    """
    # 定义文件路径
    c_o = os.path.join(lookups_loc, f'class_order_{r}.npy')
    ct = os.path.join(lookups_loc, f'counts_{r}.npy')
    b_o = os.path.join(lookups_loc, f'bucket_order_{r}.npy')
    
    # 检查查找表文件是否已存在
    if os.path.exists(c_o) and os.path.exists(ct) and os.path.exists(b_o):
        print('init lookups exists')
    else:
        # 初始化张量
        counts = torch.zeros(B + 1, dtype=torch.int32)
        bucket_order = torch.zeros(n_classes, dtype=torch.int32)
        
        # 分配桶
        for i in range(n_classes):
            bucket = mmh3(str(i), seed=r) % B
            bucket_order[i] = bucket
            counts[bucket + 1] += 1
        
        # 计算累积分布
        counts = torch.cumsum(counts, dim=0)
        
        # 初始化滚动计数和类别顺序
        rolling_counts = torch.zeros(B, dtype=torch.int32)
        class_order = torch.zeros(n_classes, dtype=torch.int32)
        
        for i in range(n_classes):
            temp = bucket_order[i].item()
            class_order[counts[temp].item() + rolling_counts[temp].item()] = i
            rolling_counts[temp] += 1
        
        # 将张量转换为 NumPy 数组
        class_order_np = class_order.numpy()
        counts_np = counts.numpy()
        bucket_order_np = bucket_order.numpy()
        
        # 保存为 .npy 文件
        np.save(c_o, class_order_np)
        np.save(ct, counts_np)
        np.save(b_o, bucket_order_np)
        print(f"Created {c_o}, {ct}, {b_o}")


# 待修复：process_scores 函数
def process_scores(inp):
    R = inp.shape[0]
    topk = inp.shape[2]
    # scores = {}
    freqs = {}
    for r in range(R):
        for k in range(topk):
            val = inp[r, 0, k]  # inp[r, 0, k] 是值，inp[r, 1, k] 是索引
            for key in inv_lookup[r, counts[r, int(inp[r, 1, k])]:counts[r, int(inp[r, 1, k]) + 1]]:
                if key in freqs:
                    # scores[key] += val
                    freqs[key] += 1  
                else:
                    # scores[key] = val
                    freqs[key] = 1
    i = 0
    while True:
        candidates = np.array([key for key in freqs if freqs[key] >= args.mf - i])
        if len(candidates) >= 10:
            break
        i += 1
    return candidates
