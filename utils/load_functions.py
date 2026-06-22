import os

import torch

from model_list import model_to_device
from .get_functions import get_save_path

__all__ = ['load_model', 'load_state_dict_from_checkpoint', 'extract_checkpoint_state_dict']

CHECKPOINT_STATE_KEYS = ('model_state_dict', 'state_dict', 'network', 'model')
STATE_DICT_PREFIXES = (
    'module.',
    'model.',
    'module.model.',
    'network.',
    'trufor.',
    'module.trufor.',
)


def _is_raw_state_dict(value):
    if not isinstance(value, dict) or not value:
        return False
    if 'param_groups' in value and 'state' in value:
        return False
    return any(torch.is_tensor(tensor) for tensor in value.values())


def extract_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in CHECKPOINT_STATE_KEYS:
            value = checkpoint.get(key)
            if _is_raw_state_dict(value):
                return value, key
        if _is_raw_state_dict(checkpoint):
            return checkpoint, '<root>'

    available_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint).__name__
    raise KeyError(
        "No model state dict found in checkpoint. Expected one of {} or a raw state_dict, got {}.".format(
            CHECKPOINT_STATE_KEYS,
            available_keys,
        )
    )


def _best_state_dict_for_target(state_dict, target_keys):
    variants = [state_dict]
    for prefix in STATE_DICT_PREFIXES:
        variants.append({
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()
        })

    target_keys = set(target_keys)
    return max(variants, key=lambda item: len(target_keys.intersection(item.keys())))


def load_state_dict_from_checkpoint(model, checkpoint):
    state_dict, checkpoint_key = extract_checkpoint_state_dict(checkpoint)
    target_model = model.module if hasattr(model, 'module') else model
    state_dict = _best_state_dict_for_target(state_dict, target_model.state_dict().keys())

    matched_keys = set(target_model.state_dict().keys()).intersection(state_dict.keys())
    if not matched_keys:
        raise KeyError(
            "Checkpoint state dict key '{}' did not match any model parameters after prefix normalization.".format(
                checkpoint_key
            )
        )

    load_result = target_model.load_state_dict(state_dict, strict=False)
    print(
        "Loaded {} tensors from checkpoint key '{}'.".format(
            len(matched_keys),
            checkpoint_key,
        )
    )
    if load_result.missing_keys:
        print("Missing keys while loading checkpoint: {}.".format(len(load_result.missing_keys)))
    if load_result.unexpected_keys:
        print("Unexpected keys while loading checkpoint: {}.".format(len(load_result.unexpected_keys)))
    return load_result


def load_model(args, model) :
    model_dirs = get_save_path(args)

    if hasattr(args, 'current_fold'):
        weight_name = 'model_weight_EPOCH{}_fold{}.pth.tar'.format(args.final_epoch, args.current_fold)
    else:
        weight_name = 'model_weight_EPOCH{}.pth.tar'.format(args.final_epoch)
    load_path = os.path.join(model_dirs, 'model_weights', weight_name)

    print("Your model is loaded from {}.".format(load_path))
    checkpoint = torch.load(load_path, map_location=getattr(args, 'device', None))
    print(".pth.tar keys() =  {}.".format(checkpoint.keys()))

    load_state_dict_from_checkpoint(model, checkpoint)

    model = model_to_device(args, model)

    return model
