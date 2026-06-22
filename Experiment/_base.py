import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CASIAV1Dataset
from model_list import model_generator, model_to_device
from utils import get_device, get_optimizer


def seed_dataloader_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class BaseSegmentationExperiment(object):
    def __init__(self, args):
        super(BaseSegmentationExperiment, self).__init__()

        self.args = args
        self.args.device = get_device(self.args)
        self.amp_enabled = torch.cuda.is_available() and not getattr(self.args, "disable_amp", False)

        self.test_loader = self.dataloader_generator()

        print("STEP2. Load 2D Image Segmentation Model {}...".format(self.args.model_name))
        self.model = model_generator(args)
        self.model = model_to_device(args, self.model)

        self.optimizer = get_optimizer(args, self.model)

    def forward(self, data):
        data = self.cpu_to_gpu(data)
        with self.autocast():
            return self.model(data)

    def autocast(self):
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=self.amp_enabled)
        return torch.cuda.amp.autocast(enabled=self.amp_enabled)

    def cpu_to_gpu(self, data):
        for key in ['image', 'target']:
            data[key] = data[key].to(self.args.device)
        return data

    def print_params(self):
        print("\ntrain data type : {}".format(self.args.train_data_type))
        print("test data type : {}".format(self.args.test_data_type))
        print("model : {}".format(self.args.model_name))
        print("optimizer : {}".format(self.optimizer))
        print("learning rate : {}".format(self.args.lr))
        print("final epoch : {}".format(self.args.final_epoch))
        print("test batch size : {}".format(self.args.test_batch_size))
        print("image size : ({}, {}, {})".format(self.args.image_size, self.args.image_size, self.args.num_channels))
        print("pytorch_total_params : {}".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def dataloader_generator(self):
        test_image_transform, test_target_transform = self.transform_generator('test')

        print("Load {} Test Dataset Loader...".format(self.args.test_data_type))
        if self.args.test_data_type != 'CASIAv1':
            raise ValueError("Wrong test data type: {}".format(self.args.test_data_type))

        test_dataset = CASIAV1Dataset(
            self.args,
            mode='test',
            transform=test_image_transform,
            target_transform=test_target_transform,
        )
        return DataLoader(
            test_dataset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_dataloader_worker,
        )
