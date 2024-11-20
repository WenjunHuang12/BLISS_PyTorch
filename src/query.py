import torch
import time
import numpy as np
import argparse
import os, sys
import traceback
import pdb
import logging
 
from net import MyModule
from dataPrepare import *
from config import config
from utils import *
from multiprocessing import Pool
sys.path.append('InvertedIndex/')
import scoreAgg

parser = argparse.ArgumentParser()
parser.add_argument("--topm", default=15, type=int)
parser.add_argument("--mf", default=2, type=int)
parser.add_argument("--gpu", default='0', type=str)
# parser.add_argument("--index", default='deep-1b_epc20_K2_B65536_R4', type=str)
parser.add_argument("--index", default='glove_epc20_K2_B4096_R4', type=str)
# parser.add_argument("--CppInf", default=1, type=bool)
parser.add_argument("--memmap", default=False, type=bool)
parser.add_argument("--rerank", default=True, type=bool)
args = parser.parse_args()

datasetName = args.index.split('_')[0]
eval_epoch = int(args.index.split('_')[1].split('epc')[1])
K = int(args.index.split('_')[2].split('K')[1])
B = int(args.index.split('_')[3].split('B')[1])
R = int(args.index.split('_')[4].split('R')[1])
feat_dim = config.DATASET[datasetName]['d']
N = config.DATASET[datasetName]['N']
metric = config.DATASET[datasetName]['metric']
dtype = config.DATASET[datasetName]['dt']
lookups_loc = "../indices/{}/".format(datasetName) + '/epoch_' + str(eval_epoch)
model_loc = "../indices/{}/".format(datasetName)
data_loc = "../../data/{}/".format(datasetName)
buffer = 1024 * (int(2 * R * N * args.topm / (B * args.mf)) // 1024)

batch_size = 32 # 查询的个数（疑似是）

# logfile = '../logs/' + datasetName + '/' + args.index + 'query.txt'
# 生成日志文件路径（使用时间戳避免覆盖）
logfile = f'../logs/{datasetName}/{args.index}_query_{time.strftime("%Y%m%d_%H%M%S")}.txt'
output_loc = logfile[:-3] + 'npy'

if args.gpu != 'all':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logfile),
        logging.StreamHandler(sys.stdout)
    ]
)

# 记录脚本命令
logging.info(f"Running script with the following parameters:")
logging.info(f"Dataset: {datasetName}, Epoch: {eval_epoch}, K: {K}, B: {B}, R: {R}, TopM: {args.topm}, MF: {args.mf}")


############################## 加载模型和索引 ################################
logging.info("Loading model...")
Model = MyModule(R)

# Check if CUDA is available and move model to GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Device: {device}")
Model.load([model_loc + '/r_' + str(r) + '_epoch_' + str(eval_epoch) + '.npz' for r in range(R)])
Model.to(device)  # Move model to GPU if available
Model.eval()
logging.info("Model loaded successfully.")

inv_lookup = np.zeros(R * N, dtype=np.int32)
counts = np.zeros(R * (B + 1), dtype=np.int32)
for r in range(R):
    inv_lookup[r * N: (r + 1) * N] = np.load(lookups_loc + '/class_order_' + str(r) + '.npy')  # block size
    counts[r * (B + 1): (r + 1) * (B + 1)] = np.load(lookups_loc + '/counts_' + str(r) + '.npy')[:B + 1]
inv_lookup = np.ascontiguousarray(inv_lookup, dtype=np.int32)
counts = np.ascontiguousarray(counts, dtype=np.int32)

fastIv = scoreAgg.PyFastIV(R, N, (B + 1), args.mf, args.topm, inv_lookup, counts)
# print("Deserialized")
logging.info("Index loaded and deserialized.")

################# 数据加载器 ####################
logging.info("Loading queries...")
[queries, neighbors100] = getQueries(datasetName)
queries = queries[:1000, :]
# print("queries loaded ", queries.shape)
logging.info(f"Queries loaded. Shape: {queries.shape}")

# 将查询转换为 PyTorch 张量
queries_tensor = torch.from_numpy(queries).float()
dataset_queries = torch.utils.data.TensorDataset(queries_tensor)
dataloader = torch.utils.data.DataLoader(dataset_queries, batch_size=batch_size)

if args.rerank:
    datapath = data_loc + 'fulldata.dat'
    dataset = getFulldata(datasetName, datapath)
    if metric == "L2":
        norms = np.load(data_loc + "norms.npy")
    if metric == "cosine":
        norms = np.load(data_loc + "norms.npy")
        dataset = dataset / (norms[:, None])
    # print("dense vectors loaded")
    logging.info("Dense vectors loaded.")

count = 0
score_sum = [0.0, 0.0, 0.0]
output = -1 * np.ones([10000, 10])

bthN = 0
begin_time = time.time()

Inf = 0
RetRank = 0

start_time = time.time()

for x_batch_tuple in dataloader:
    try:
        x_batch = x_batch_tuple[0].to(device)  # Move batch to GPU
        t1 = time.time()
        with torch.no_grad():
            top_buckets_ = Model(x_batch, args.topm)  # 应返回 [R, batch_size, topm]

        # 将 top_buckets_ 转换为张量并移动到 CPU
        top_buckets_ = torch.stack(top_buckets_, dim=0).cpu().numpy()  # [R, batch_size, topm]
        top_buckets_ = np.transpose(top_buckets_, (1, 0, 2))  # [batch_size, R, topm]

        # 添加调试信息
        logging.info("Top buckets shape:", top_buckets_.shape)
        logging.info("Top buckets sample:", top_buckets_[0])

        len_cands = np.zeros(top_buckets_.shape[0])
        t2 = time.time()
        Inf += (t2 - t1)
        i = 0  # 确保 i 在异常处理中可访问
        for i in range(top_buckets_.shape[0]):
            candid = np.empty(buffer, dtype='int32')
            candSize = np.empty(1, dtype='int32')
            fastIv.FC(np.ascontiguousarray(top_buckets_[i, :, :], dtype=np.int32).reshape(-1), buffer, candid, candSize)
            candidates = candid[0: candSize[0]]

            score_sum[1] += len(candidates)
            if args.rerank:
                x_batch_i = x_batch[i].cpu().numpy()
                if metric == "IP":
                    dists = np.dot(dataset[candidates], x_batch_i)
                if metric == "L2":
                    dists = 2 * np.dot(dataset[candidates], x_batch_i) - norms[candidates]
                if metric == "cosine":
                    dists = np.dot(dataset[candidates], x_batch_i)
                if len(dists) <= 10:
                    output[bthN * batch_size + i, :len(dists)] = candidates
                if len(dists) > 10:
                    top_cands = np.argpartition(dists, -10)[-10:]
                    output[bthN * batch_size + i, :10] = candidates[top_cands]
                    candidates = candidates[top_cands]

            score_sum[0] += len(np.intersect1d(candidates, neighbors100[bthN * batch_size + i, :10])) / 10

        t3 = time.time()
        RetRank += t3 - t2
        bthN += 1
        print(bthN)

        # Calculate QPS
        queries_processed = bthN * batch_size
        time_elapsed = t3 - start_time
        qps = queries_processed / time_elapsed  # queries per second
        logging.info(f"Processed {queries_processed} queries in {time_elapsed:.2f} seconds. QPS: {qps:.2f}")

    except Exception as e:
        # print("An exception occurred:", e)
        logging.error("An exception occurred: %s", e)
        traceback.print_exc()

        # Log exception details
        total_points = bthN * batch_size
        if total_points > len(neighbors100):
            total_points = len(neighbors100)
        logging.error(f'Overall Recall for {total_points} points: {score_sum[0] / total_points}')
        logging.error(f'Avg candidate size for {total_points} points: {score_sum[1] / total_points}')
        logging.error(f'Inf per point: {Inf / ((bthN - 1) * batch_size)}')
        logging.error(f'Ret+rank per point: {RetRank / ((bthN - 1) * batch_size)}')
        logging.error(f'Per point to report: {(Inf / 32 + RetRank / 4) / ((bthN - 1) * batch_size)}')

        np.save(output_loc, output)
        break

# Final summary
end_time = time.time()
total_time = end_time - start_time
logging.info(f"Total time for processing {queries_processed} queries: {total_time:.2f} seconds")

total_points = bthN * batch_size
if total_points > len(neighbors100):
    total_points = len(neighbors100)
logging.info(f'Overall Recall for {total_points} points: {score_sum[0] / total_points}')
logging.info(f'Avg candidate size for {total_points} points: {score_sum[1] / total_points}')
logging.info(f'Inf per point: {Inf / ((bthN - 1) * batch_size)}')
logging.info(f'Ret+rank per point: {RetRank / ((bthN - 1) * batch_size)}')
logging.info(f'Per point to report: {(Inf / 32 + RetRank / 4) / ((bthN - 1) * batch_size)}')