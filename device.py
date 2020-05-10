import torch

# GPU使用か否か
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
