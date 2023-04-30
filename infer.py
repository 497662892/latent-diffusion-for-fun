import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import DataLoader
import json
import torchaudio
import random
from torchvision.transforms import Resize
from config import data_path, dataset2wavpath

import extract_sp
import extract_mcep

from ldm.data.singvoice import SingVoice

#get the input of source audio 




def get_uids(dataset, dataset_type):
    dataset_file = os.path.join(data_path, dataset, "{}.json".format(dataset_type))
    with open(dataset_file, "r") as f:
        utterances = json.load(f)

    if dataset == "M4Singer":
        uids = ["{}_{}_{}".format(u["Singer"], u["Song"], u["Uid"]) for u in utterances]
        upaths = [u["Path"] for u in utterances]
        return uids, upaths
    else:
        return [u["Uid"] for u in utterances]


def save_audio(path, waveform, fs):
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if len(waveform.size()) == 1:
        waveform = waveform[None, :]
    # print("HERE: waveform", waveform.shape, waveform.dtype, waveform)
    torchaudio.save(path, waveform, fs, encoding="PCM_S", bits_per_sample=16)


def save_pred_audios(
    pred, args, index, uids, output_dir):
    dataset = args.dataset
    wave_dir = dataset2wavpath[dataset]


    # Predict
    
    sample_uids = uids[index]

    for uid in sample_uids:
        wave_file = os.path.join(wave_dir, "{}.wav".format(uid))
        f0, sp, ap, fs = extract_sp.extract_world_features(wave_file)
        frame_len = len(f0)

        mcep = pred[:frame_len]
        sp_pred = extract_mcep.mgc2SP(mcep)
        assert sp.shape == sp_pred.shape

        y_gt = extract_sp.world_synthesis(f0, sp, ap, fs)
        y_pred = extract_sp.world_synthesis(f0, sp_pred, ap, fs)

        # save gt
        gt_file = os.path.join(output_dir, "{}_gt.wav".format(uid))
        os.system("cp {} {}".format(wave_file, gt_file))
        # save WORLD synthesis gt
        world_gt_file = os.path.join(output_dir, "{}_gt_world.wav".format(uid))
        save_audio(world_gt_file, y_gt, fs)
        # save WORLD synthesis pred
        world_pred_file = os.path.join(
            output_dir,
            "{}_pred_world.wav".format(uid)
        )
        save_audio(world_pred_file, y_pred, fs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="M4Singer",
        help="the name of the testing dataset",
    )
    opt = parser.parse_args()
    
    data_path = opt.indir
    
    test_dataset = SingVoice(data_path, "M4Singer", "test")
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    uids = get_uids(opt.dataset, "test")
    
    config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("models/ldm/singingvoice/last.ckpt")["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    
    with torch.no_grad():
        with model.ema_scope():
            for i, cases in enumerate(tqdm(test_dataloader)):
                
                mask, whisper, f0 =  cases["mask"], cases["whisper"], cases["f0"]
                # encode masked image and concat downsampled mask
                mask = mask.bool()
                mask_post = Resize((mask.shape[1]//4,1))(mask) #downsample mask [B,T//4,1]
                
                c = {"whisper": whisper, "f0": f0, "mask": mask_post}
                shape = (1, 3, 200, 10)
                
                #sampling
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)
                
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                #get the predicted mcep
                pred_mcep = x_samples_ddim[0][0] #take the first channel
                
                #turn the predicted mcep to sp
                save_pred_audios(pred_mcep, opt, index=i, uids=uids, output_dir=opt.outdir)






