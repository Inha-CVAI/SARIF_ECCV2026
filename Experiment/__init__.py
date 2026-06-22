import os

from model_list import _training_configuration
from utils import catnetv2_adjust_learning_rate
from .forgery_detection_experiment import ForgeryDetectionExperiment


DEFAULT_MIXED_TRAIN_DATA_TYPE = "CASIAv2_IMD2020_FantasticReality_TampCOCO"

def apply_training_schema(args):
    if getattr(args, 'training_schema', 'model_default') == 'CATNetV2':
        args.optimizer_name = 'SGD'
        args.lr = 0.005
        args.momentum = 0.9
        args.nesterov = False
        args.weight_decay = 1e-4
        args.final_epoch = 30
        args.training_epochs = 30
        args.adjust_learning_rate = catnetv2_adjust_learning_rate
        args.select_best_epoch = False
        args.best_metric_name = 'DSC'
    return args

def dataset_configuration(args):
    best_epoch = getattr(args, 'best_epoch', None)
    checkpoint_epoch = getattr(args, 'checkpoint_epoch', None)
    if getattr(args, 'train_data_type', None) is None:
        args.train_data_type = DEFAULT_MIXED_TRAIN_DATA_TYPE
    if getattr(args, 'test_data_type', None) is None and not getattr(args, 'train', False):
        args.test_data_type = getattr(args, 'valid_data_type', None) or 'CoMoFoD'

    args.train_dataset_dir = os.path.join(args.data_path, args.train_data_type)
    args.test_dataset_dir = os.path.join(args.data_path, args.test_data_type) if args.test_data_type else None

    args = _training_configuration(args)
    args = apply_training_schema(args)
    if not args.train and best_epoch is not None:
        args.final_epoch = best_epoch
    elif not args.train and checkpoint_epoch is not None:
        args.final_epoch = checkpoint_epoch
    return args
