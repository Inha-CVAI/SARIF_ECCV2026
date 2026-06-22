import importlib

import torch


MODEL_REGISTRY = {
    "SARIF": ("model_list.SARIF", "Model"),
    "sarif": ("model_list.SARIF", "Model"),
}

TRAINING_CONFIG_REGISTRY = {
    "SARIF": ("model_list.SARIF", "_training_config"),
    "sarif": ("model_list.SARIF", "_training_config"),
}


def _load_registry_attr(registry, name):
    if name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError("Wrong model name '{}'. Available models: {}".format(name, available))

    module_name, attr_name = registry[name]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def model_generator(args):
    model_cls = _load_registry_attr(MODEL_REGISTRY, args.model_name)
    return model_cls(args)


def _training_configuration(args):
    training_config = _load_registry_attr(TRAINING_CONFIG_REGISTRY, args.model_name)
    return training_config(args)


def model_to_device(args, model):
    return model.to(args.device)


def pos_weight_calculator(label_mask):
    pos_weight = []

    for target in label_mask:
        num_pos = (target == 1).sum().item()
        num_neg = (target == 0).sum().item()
        pos_weight.append(num_neg / (num_pos + 1e-6))

    return torch.tensor(pos_weight).to(label_mask.device).view(-1, 1, 1, 1)
