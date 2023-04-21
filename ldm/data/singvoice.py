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
import librosa


class SingVoice(Dataset):
    def __init__(self, data_path, dataset, dataset_type,padding_size = 800):
        self.dataset_type = dataset_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding_size = padding_size
        self.dataset_dir = os.path.join(data_path, dataset)

        logging.info("\n" + "=" * 20 + "\n")
        logging.info("{} Dataset".format(dataset_type))
        self.loading_data()
        logging.info("\n" + "=" * 20 + "\n")
        

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
            (sz, self.padding_size, self.y_d), device=self.device, dtype=torch.float
        )
        self.y_mask = torch.zeros(
            (sz, self.padding_size, 1), device=self.device, dtype=torch.long
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
            print('the shape of f0 is', len(self.f0)) #for debug
            
            
        logging.info(
            "f0: sz = {}, shape = {}".format(
                len(self.whisper), self.f0[0].shape
            )
        )
        

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
        whisper = self.whisper[idx]
        f0 = self.f0[idx]
        
        self.whisper_dim = self.whisper[0].shape[1]
        whisper_gt = torch.zeros((self.padding_size, self.whisper_dim), device=self.device, dtype=torch.float)
        f0_gt = torch.zeros((self.padding_size, 1), device=self.device, dtype=torch.float)
        
        sz = min(self.padding_size, len(whisper))
        
        whisper_gt[:sz] = torch.as_tensor(whisper[:sz], device=self.device)
        f0_gt = torch.as_tensor(f0[:sz], device=self.device)
        f0_gt = get_bin_index(f0_gt) # convert to bin index
        
        sample = {"id":idx, "MECP":self.y_gt[idx], "mask":self.y_mask[idx], "f0":f0_gt, "whisper":whisper_gt}
        return sample

    def get_padding_y_gt(self, idx):
        y_gt = torch.zeros(
            (self.padding_size, self.y_d), device=self.device, dtype=torch.float
        )
        mask = torch.ones(
            (self.padding_size, 1), device=self.device, dtype=torch.long
        )

        mcep = self.mcep[idx]
        sz = min(self.padding_size, len(mcep))
        y_gt[:sz] = torch.as_tensor(mcep[:sz], device=self.device)
        mask[sz:] = 0

        return y_gt, mask



def get_bin_index(f0, n_bins=300, m = "C2", M = "C7"):
    """
    Args:
        f0: tensor whose shpae is (N, frame_len)
    Returns:
        index: tensor whose shape is same to f0
    """
    # Set normal index in [1, n_bins - 1]
    m = librosa.note_to_hz(m)
    M = librosa.note_to_hz(M)
    
    width = (M + 1e-7 - m) / (n_bins - 1)
    index = (f0 - m) // width + 1
    # Set unvoiced frames as 0
    index[torch.where(f0 == 0)] = 0
    # Therefore, the vocabulary is [0, n_bins- 1], whose size is n_bins
    return torch.as_tensor(index, dtype=torch.long, device=f0.device)






