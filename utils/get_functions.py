import os
import sys
import math

import torch
import torch.optim as optim
__all__ = ['get_device', 'get_optimizer','get_save_path', 'catnetv2_adjust_learning_rate']
def get_device(args=None):
    if torch.cuda.is_available():
        gpu = getattr(args, "gpu", None) if args is not None else None
        device = torch.device("cuda:{}".format(gpu)) if gpu is not None else torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("You are using \"{}\" device.".format(device))

    return device

def get_optimizer(args, model):
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(name)

    if args.optimizer_name == 'SGD' :
        optimizer = optim.SGD(
            params=params,
            lr=args.lr,
            momentum=getattr(args, "momentum", 0.9),
            nesterov=getattr(args, "nesterov", True),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer_name == 'Adam' : optimizer = optim.Adam(params=params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name == 'AdamW' : optimizer = optim.AdamW(params=params, lr=args.lr, weight_decay=args.weight_decay)
    else : print("Wrong optimizer"); sys.exit()

    return optimizer

def catnetv2_adjust_learning_rate(optimizer, epochs, train_loader_len, learning_rate):
    total_steps = max(1, epochs * train_loader_len)
    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)),
    )

def get_save_path(args):
    path_epoch = getattr(args, 'training_epochs', args.final_epoch)
    save_model_path = '{}_{}x{}_{}_{}({}&{})_{}({}_{})'.format(
        args.train_data_type,
        str(args.image_size),
        str(args.image_size),
        str(args.train_batch_size),
        args.model_name,
        args.cnn_backbone,
        args.transformer_backbone,
        args.optimizer_name,
        args.lr,
        str(path_epoch).zfill(3),
    )

    model_dirs = os.path.join(args.save_path, save_model_path)
    os.makedirs(os.path.join(model_dirs, 'model_weights'), exist_ok=True)
    os.makedirs(os.path.join(model_dirs, 'test_reports'), exist_ok=True)
    os.makedirs(os.path.join(model_dirs, 'test_reports_per_cases'), exist_ok=True)
    os.makedirs(os.path.join(model_dirs, 'plot_results'), exist_ok=True)

    return model_dirs
