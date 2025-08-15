from typing import List, Tuple
import torch.nn as nn


def split_decay_groups(
        named_params: List[Tuple[str, object]]
    ) -> Tuple[List[object], List[object]]:
    """avoids weight decay on LayerNorm and bias terms"""
    decay, no_decay = [], []
    for _, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    return decay, no_decay


def get_activation(activation: str, **kwargs) -> nn.Module:
    """
    Get the activation function based on the provided name.

    Parameters
    ----------
    activation : str
        Name of the activation function.
    **kwargs
        Additional keyword arguments to
        pass to the activation function.

    Returns
    -------
    nn.Module
        The corresponding activation function module.
    """
    if activation == 'ELU':
        return nn.ELU(**kwargs)
    elif activation == 'ReLU':
        return nn.ReLU(**kwargs)
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(**kwargs)
    elif activation == 'PReLU':
        return nn.PReLU(**kwargs)
    elif activation == 'GLU':
        return nn.GLU(**kwargs)
    elif activation == 'GELU':
        return nn.GELU(**kwargs)
    else:
        raise ValueError(
            f"Unsupported activation function: {activation}")
