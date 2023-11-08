# 测试各函数作用
# 对实际训练并无影响

import torch

array = torch.tensor([[1, 0], [-1, -1]], dtype=torch.float)
base = torch.tensor([1, 0], dtype=torch.float)
output = torch.cosine_similarity(base, array)
print(output)
