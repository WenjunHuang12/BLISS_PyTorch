import torch
import torch.nn as nn
import numpy as np

class MyModule(nn.Module):
    def __init__(self, R):
        super(MyModule, self).__init__()
        self.R = R
        self.layer1 = nn.ModuleList()
        self.layer2 = nn.ModuleList()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load(self, paths):
        params = [np.load(path) for path in paths]
        for r in range(self.R):
            # 加载参数并转换为 PyTorch 张量
            W1 = torch.tensor(params[r]['W1'], dtype=torch.float32, device=self.device)
            b1 = torch.tensor(params[r]['b1'], dtype=torch.float32, device=self.device)
            W2 = torch.tensor(params[r]['W2'], dtype=torch.float32, device=self.device)
            b2 = torch.tensor(params[r]['b2'], dtype=torch.float32, device=self.device)

            # 获取输入和输出特征的维度
            in_features1 = W1.shape[0]
            out_features1 = W1.shape[1]
            in_features2 = W2.shape[0]
            out_features2 = W2.shape[1]

            # 定义第一个全连接层
            layer1_r = nn.Linear(in_features1, out_features1).to(self.device)
            layer1_r.weight.data = W1.t()  # 转置权重矩阵以匹配 PyTorch 的维度
            layer1_r.bias.data = b1
            layer1_r.weight.requires_grad = False  # 如果不需要训练这些参数，可以将 requires_grad 设为 False
            layer1_r.bias.requires_grad = False

            # 定义第二个全连接层
            layer2_r = nn.Linear(in_features2, out_features2).to(self.device)
            layer2_r.weight.data = W2.t()
            layer2_r.bias.data = b2
            layer2_r.weight.requires_grad = False
            layer2_r.bias.requires_grad = False

            # 将层添加到 ModuleList 中
            self.layer1.append(layer1_r)
            self.layer2.append(layer2_r)

    def forward(self, x, topk):
        top_buckets = []
        x = x.to(self.device)
        with torch.no_grad(): # 推理阶段禁用梯度计算
            for r in range(self.R):
                # 前向传播
                hidden_layer = torch.relu(self.layer1[r](x))
                logits = self.layer2[r](hidden_layer)
                # 获取 top-k 的索引
                _, top_indices = torch.topk(logits, k=topk, dim=-1, largest=True, sorted=False)
                top_buckets.append(top_indices)
        return top_buckets
