from typing import List, Tuple


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
