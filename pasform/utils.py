def set_seed(seed):
    import random
    import numpy as np
    import torch
    import open3d as o3d

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    o3d.utility.random.seed(seed)

    return