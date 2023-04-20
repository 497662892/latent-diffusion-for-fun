from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




import pickle
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import logging
import time
import sys
from config import data_path
from extract_f0 import get_bin_index

class SingVoice(Dataset):
    def __init__(self, data_path, dataset, dataset_type,padding_size = 800):
        self.dataset_type = dataset_type

        self.dataset_dir = os.path.join(data_path, dataset)

        logging.info("\n" + "=" * 20 + "\n")
        logging.info("{} Dataset".format(dataset_type))
        self.loading_data()
        logging.info("\n" + "=" * 20 + "\n")
        self.padding_size = padding_size

    def loading_whisper(self):
        logging.info("Loading Whisper features...")
        with open(
            os.path.join(self.dataset_dir, "Whisper/{}.pkl".format(self.dataset_type)),
            "rb",
        ) as f:
            self.whisper = pickle.load(f)
        logging.info(
            "Whisper: sz = {}, shape = {}".format(
                len(self.whisper), self.whisper[0].shape
            )
        )
        self.whisper_dim = self.whisper[0].shape[1]
        #padding for whisper based on mask
        for i in range(len(self.whisper)):
            whisper = self.whisper[i]
            whisper_gt = torch.zeros((self.padding_size, self.whisper_dim), device=self.args.device, dtype=torch.float)
            sz = min(self.padding_size, len(whisper))
            whisper_gt[:sz] = torch.as_tensor(whisper[:sz], device=self.args.device)
            
        self.whisper = whisper_gt
            
            
    def loading_MCEP(self):
        logging.info("Loading MCEP features...")
        with open(
            os.path.join(self.dataset_dir, "MCEP/{}.pkl".format(self.dataset_type)),
            "rb",
        ) as f:
            self.mcep = pickle.load(f)
        logging.info(
            "MCEP: sz = {}, shape = {}".format(len(self.mcep), self.mcep[0].shape)
        )
        self.y_d = self.mcep[0].shape[1]

        # Padding
        sz = len(self.mcep)
        self.y_gt = torch.zeros(
            (sz, self.padding_size, self.y_d), device=self.args.device, dtype=torch.float
        )
        self.y_mask = torch.zeros(
            (sz, self.padding_size, 1), device=self.args.device, dtype=torch.long
        )
        for idx in range(sz):
            y, mask = self.get_padding_y_gt(idx)
            self.y_gt[idx] = y
            self.y_mask[idx] = mask
            
    def loading_f0(self):
        logging.info("Loading f0 features...")
        with open(
            os.path.join(self.dataset_dir, "f0/44100/pyin/{}.pkl".format(self.dataset_type)),
            "rb",
        ) as f:
            self.f0 = pickle.load(f)
            self.f0 = get_bin_index(self.f0) # convert to bin index
        logging.info(
            "f0: sz = {}, shape = {}".format(
                len(self.whisper), self.f0[0].shape
            )
        )
        
        for i in range(len(self.f0)):
            f0 = self.f0[i]
            f0_gt = torch.zeros((self.padding_size, 1), device=self.args.device, dtype=torch.float)
            sz = min(self.padding_size, len(f0))
            f0_gt[:sz] = torch.as_tensor(f0[:sz], device=self.args.device)
            
        self.f0 = f0_gt

    def loading_data(self):
        t = time.time()
        
        self.loading_MCEP()
        self.loading_whisper()
        self.loading_f0

        logging.info("Done. It took {:.2f}s".format(time.time() - t))

    def __len__(self):
        return len(self.y_gt)

    def __getitem__(self, idx):
        # y_gt, mask = self.get_padding_y_gt(idx)
        sample = {"id":idx, "MECP":self.y_gt[idx], "mask":self.y_mask[idx], "f0":self.f0[idx], "whisper":self.whisper[idx]}
        return sample

    def get_padding_y_gt(self, idx):
        y_gt = torch.zeros(
            (self.padding_size, self.y_d), device=self.args.device, dtype=torch.float
        )
        mask = torch.ones(
            (self.padding_size, 1), device=self.args.device, dtype=torch.long
        )

        mcep = self.mcep[idx]
        sz = min(self.padding_size, len(mcep))
        y_gt[:sz] = torch.as_tensor(mcep[:sz], device=self.args.device)
        mask[sz:] = 0

        return y_gt, mask










