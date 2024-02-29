import os
import torch
import numpy as np
import random

def collect_file(file_path, match='.h5'):
    file_list = []
    for root, dir, files in os.walk(file_path):
        for f in files:
            if match.lower() in f.lower():
                file_list.append(os.path.join(root, f))
    return file_list

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True