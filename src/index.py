import torch
import time
import numpy as np
import argparse
import os
import pdb
import sys
import logging
from dataPrepare import *
from config import config
from net import MyModule

parser = argparse.ArgumentParser()
parser.add_argument("--index", default='glove_epc20_K2_B4096_R4', type=str)
args = parser.parse_args()
datasetName = args.index.split('_')[0]  
n_epochs = int(args.index.split('_')[1].split('epc')[1]) 
K = int(args.index.split('_')[2].split('K')[1])  
B = int(args.index.split('_')[3].split('B')[1])
R = int(args.index.split('_')[4].split('R')[1])

def Index(B, R, datasetName, load_epoch, K):
    bucketSort = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 如果不使用所有 GPU，可以指定特定的 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    ######################### 修改部分
    model_save_loc = "../indices/{}/".format(datasetName)
    lookups_loc  = "../indices/{}/".format(datasetName)
    N = config.DATASET[datasetName]['N'] 
    train_data_loc = "../../data/{}/".format(datasetName)
    batch_size = 5000
    # batch_size = 2000

    # 日志记录配置
    log_filename = os.path.join(lookups_loc, f'index_log_{time.strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO)

    # 记录执行的命令行和设备信息
    logging.info(f'Executing command: {" ".join(sys.argv)}')
    logging.info(f'Using device: {device}')

    # 加载模型
    Model = MyModule(R)
    Model.load([model_save_loc + '/r_' + str(r) + '_epoch_' + str(load_epoch) + '.npz' for r in range(R)])
    Model.to(device)
    Model.eval()  # 设置模型为评估模式
    # print("model loaded")
    logging.info("Model loaded successfully.")

    # 加载数据并创建 DataLoader
    datapath = train_data_loc + '/fulldata.dat'
    full_data = getFulldata(datasetName, datapath).astype(np.float32)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(full_data))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    # print("data loaded")
    logging.info("Data loaded successfully.")

    top_preds = np.zeros([R, N, K], dtype=np.int32)

    inference_start_time = time.time()

    start_idx = 0
    for batch in dataloader:
        inputs = batch[0]  # 获取输入数据
        batch_size_actual = inputs.size(0)
        # 将输入数据移动到设备（如 GPU）上
        inputs = inputs.to(device)
        # 前向传播
        with torch.no_grad():
            outputs = Model(inputs, K)
        # 将结果转换为 NumPy 数组
        outputs_np = np.array([output.cpu().numpy() for output in outputs])  # outputs 是一个列表
        top_preds[:, start_idx:start_idx+batch_size_actual] = outputs_np
        start_idx += batch_size_actual

        # sys.stdout.write("Inference progress: %d%%   \r" % (start_idx * 100 / N))
        # sys.stdout.flush()
        logging.info(f"Inference progress: {start_idx * 100 / N:.2f}%")

    if start_idx < N:
        # print(start_idx)
        # assert (start_idx >= N), "batch iterator issue!"
        logging.error(f"Batch iterator issue! Only processed {start_idx} out of {N} samples.")
        assert start_idx >= N, "batch iterator issue!"

    inference_end_time = time.time()
    # print("Inference time: ", inference_end_time - inference_start_time)
    logging.info(f"Inference completed. Total inference time: {inference_end_time - inference_start_time:.2f} seconds.")

    #####################################
    try:
        indexing_start_time = time.time()

        # 可以考虑并行化处理
        for r in range(R):
            counts = np.zeros(B+1, dtype=np.int32)
            bucket_order = np.zeros(N, dtype=np.int32)
            for i in range(N):
                bucket = top_preds[r, i, np.argmin(counts[top_preds[r, i] + 1])] 
                bucket_order[i] = bucket
                counts[bucket + 1] += 1  

            counts = np.cumsum(counts)
            class_order = np.zeros(N, dtype=np.int32)
            class_order = np.argsort(bucket_order)
            # 对桶进行排序
            if bucketSort:
                for b in range(B):
                    class_order[counts[b]:counts[b+1]] = np.sort(class_order[counts[b]:counts[b+1]])
            ###
            folder_path = lookups_loc + 'epoch_' + str(load_epoch)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            np.save(folder_path + '/class_order_' + str(r) + '.npy', class_order)
            np.save(folder_path + '/counts_' + str(r) + '.npy', counts)
            np.save(folder_path + '/bucket_order_' + str(r) + '.npy', bucket_order)
            # print(r)
            logging.info(f"Processed bucket {r}.")
    except Exception as e:
        # print("check indexing issue", r)
        # print(e)
        logging.error(f"Error during indexing: {e}")
        raise
    # index_and_save_time = time.time()
    # print("indexed and saved in time: ", index_and_save_time - inference_end_time)
    indexing_end_time = time.time()
    logging.info(f"Indexing completed. Total indexing time: {indexing_end_time - indexing_start_time:.2f} seconds.")
    
    total_time = indexing_end_time - inference_start_time
    logging.info(f"Total process time: {total_time:.2f} seconds.")

Index(B, R, datasetName, n_epochs, K)
