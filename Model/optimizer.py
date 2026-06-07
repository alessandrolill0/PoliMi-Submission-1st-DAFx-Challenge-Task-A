import torch.optim as optim



def get_optimizer(active_params ,lr: float = 0.01):
    # Single LR across all parameters.
    return optim.Adam(active_params, lr=lr)

