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


def load_state_dict(
        model: nn.Module, state_dict: dict,
        prefix: str='',
        ignore_missing: List[str]=[],
        verbose: bool=True
    ):
    """
    Load a state_dict into a model, handling missing and unexpected keys.

    Parameters
    ----------
    model : torch.nn.Module
        The model into which the state_dict will be loaded.
    state_dict : dict
        The state_dict containing the weights to load.
    prefix : str, optional
        A prefix to prepend to all keys in the state_dict, by default ''.
    ignore_missing : List[str], optional
        A list of substrings to ignore when checking for missing keys,
        by default [].
    """
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing:
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if verbose:
        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(ignore_missing_keys) > 0:
            print("Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys))
        if len(error_msgs) > 0:
            print('\n'.join(error_msgs))


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
