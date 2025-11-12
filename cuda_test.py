import torch

# 验证PyTorch是否安装成功
print(torch.__version__)  # 输出PyTorch版本，如2.2.0

# 验证CUDA是否可用
print(torch.cuda.is_available())  # 输出True表示GPU可用，False表示不可用

# 验证GPU数量和型号
print(torch.cuda.device_count())  # 输出GPU数量（如1）
print(torch.cuda.get_device_name(0))  # 输出GPU型号（如NVIDIA GeForce RTX 3060）
