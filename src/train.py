import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os
import numpy as np
import logging
from dataPrepare import *
from utils import *

# gpu_usage is not used now
def trainIndex(lookups_loc, train_data_loc, datasetName, model_save_loc, batch_size, B, vec_dim, hidden_dim, logfile,
              r, gpu, gpu_usage, load_epoch, k2, n_epochs):

    # 检查并准备训练数据
    getTraindata(datasetName)  # 检查数据是否存在，确保 ground truth 正确
    
    logging.info(f"Running script with the following parameters:")
    logging.info(f"Dataset: {datasetName}, Batch size: {batch_size}, K: {k2}, B: {B}, epoch: {n_epochs}")

    # 设置设备
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # 加载训练数据
    x_train = np.load(os.path.join(train_data_loc, 'train.npy'))  # 形状：[N, vec_dim]
    y_train = np.load(os.path.join(train_data_loc, 'groundTruth.npy'))  # 形状：[N, 100]
    N = x_train.shape[0]

    epoch_dir = os.path.join(lookups_loc, f'epoch_{load_epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    create_universal_lookups(r, B, N, epoch_dir)

    # 加载 lookup
    lookup_path = os.path.join(epoch_dir, f'bucket_order_{r}.npy')
    lookup = np.load(lookup_path)[:N]
    lookup_tensor = torch.from_numpy(lookup).long().to(device)

    # 定义模型
    class MyModel(nn.Module):
        def __init__(self, vec_dim, hidden_dim, B):
            super(MyModel, self).__init__()
            self.fc1 = nn.Linear(vec_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, B)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            logits = self.fc2(x)
            return logits

    model = MyModel(vec_dim, hidden_dim, B).to(device)

    # 如果 load_epoch > 0，则加载模型参数
    if load_epoch > 0:
        params_path = os.path.join(model_save_loc, f'r_{r}_epoch_{load_epoch}.npz')
        params = np.load(params_path)
        with torch.no_grad():
            model.fc1.weight.copy_(torch.from_numpy(params['W1'].T))
            model.fc1.bias.copy_(torch.from_numpy(params['b1']))
            model.fc2.weight.copy_(torch.from_numpy(params['W2'].T))
            model.fc2.bias.copy_(torch.from_numpy(params['b2']))
        del params

    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    # 创建数据加载器
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True)

    # 日志记录配置
    logfile_path = os.path.join(logfile, f'logs_{r}_{time.strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(filename=logfile_path, level=logging.INFO)

    # begin_time = time.time()
    total_start_time = time.time()
    total_time = 0
    n_check = 1000
    n_steps_per_epoch = N // batch_size

    for curr_epoch in range(load_epoch + 1, load_epoch + n_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        count = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            step_start_time = time.time()
            data = data.to(device)  # 输入数据，形状：[batch_size, vec_dim]
            target = target.to(device)  # 标签，形状：[batch_size, 100]
            batch_size_actual = data.size(0)

            # 构建标签 y_
            with torch.no_grad():
                batch_y_flat = target.view(-1)  # 形状：[batch_size_actual * 100]
                batch_y_mapped = lookup_tensor[batch_y_flat]  # 形状：[batch_size_actual * 100]

                y_ = torch.zeros(batch_size_actual, B, device=device)
                batch_indices = torch.arange(batch_size_actual, device=device).unsqueeze(1).repeat(1, 100).view(-1)  # 形状：[batch_size_actual * 100]
                y_[batch_indices, batch_y_mapped] = 1

            # 前向传播
            logits = model(data)  # 形状：[batch_size_actual, B]

            # 计算损失
            loss = criterion(logits, y_)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += 1
            if count % n_check == 0:
                # 记录日志信息
                step_time = time.time() - step_start_time
                total_time += step_time
                logging.info(f'Finished {count} steps. Time elapsed for last {n_check} steps: {step_time:.2f} s')
                logging.info(f'Train loss: {loss.item()}')

        # 记录 epoch 结束信息
        epoch_time = time.time() - epoch_start_time
        logging.info('###################################')
        logging.info(f'Finished epoch {curr_epoch} in {epoch_time:.2f} seconds')
        logging.info(f'Total time elapsed so far: {total_time:.2f} seconds')
        logging.info('###################################')

        # 每 5 个 epoch 保存一次模型参数和更新 lookup
        if curr_epoch % 5 == 0:
            # 保存模型参数到 .npz 文件
            params = {
                'W1': model.fc1.weight.detach().cpu().numpy().T,
                'b1': model.fc1.bias.detach().cpu().numpy(),
                'W2': model.fc2.weight.detach().cpu().numpy().T,
                'b2': model.fc2.bias.detach().cpu().numpy(),
            }
            params_path = os.path.join(model_save_loc, f'r_{r}_epoch_{curr_epoch}.npz')
            np.savez_compressed(params_path, **params)
            del params

            # 计算整个数据集的预测结果，获取 top_k 预测
            model.eval()
            top_preds_list = []
            with torch.no_grad():
                for i in range(0, N, batch_size):
                    batch_data = x_train_tensor[i:i + batch_size].to(device)
                    batch_logits = model(batch_data)  # [batch_size, B]
                    _, top_batch_preds = torch.topk(batch_logits, k=k2, dim=1, largest=True, sorted=True)  # [batch_size, k2]
                    top_preds_list.append(top_batch_preds.cpu())

            top_preds = torch.cat(top_preds_list, dim=0).numpy()  # [N, k2]
            logging.info(f"Top-k prediction time: {time.time() - epoch_start_time:.2f} seconds")

            # 更新 lookup
            counts = np.zeros(B + 1, dtype=int)
            bucket_order = np.zeros(N, dtype=int)
            for i in range(N):
                # 找到 top_k 中 counts 最小的桶
                possible_buckets = top_preds[i]
                bucket_min = possible_buckets[np.argmin(counts[possible_buckets + 1])]
                bucket_order[i] = bucket_min
                counts[bucket_min + 1] += 1

            # 更新 lookup_tensor
            lookup = torch.from_numpy(bucket_order).long().to(device)
            lookup_tensor = lookup

            # 计算 class_order
            counts = np.cumsum(counts)
            rolling_counts = np.zeros(B, dtype=int)
            class_order = np.zeros(N, dtype=int)
            for i in range(N):
                temp = bucket_order[i]
                class_order[counts[temp] + rolling_counts[temp]] = i
                rolling_counts[temp] += 1

            # 可选：保存 class_order、counts、bucket_order
            # folder_path = os.path.join(lookups_loc, f'epoch_{curr_epoch}')
            # os.makedirs(folder_path, exist_ok=True)
            # np.save(os.path.join(folder_path, f'class_order_{r}.npy'), class_order)
            # np.save(os.path.join(folder_path, f'counts_{r}.npy'), counts)
            # np.save(os.path.join(folder_path, f'bucket_order_{r}.npy'), bucket_order)

            # begin_time = time.time()

    # 记录总的训练时间
    total_time = time.time() - total_start_time
    logging.info(f'Training finished. Total time elapsed: {total_time:.2f} seconds')