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
import torch.nn as nn
from ldm.data.singvoice import SingVoice

sys.path.append("../")
from config import data_path, dataset2wavpath
sys.path.append("../preprocess")
import extract_sp
import extract_mcep

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
    pred,mecp_get, args, index, uids, output_dir, upaths = None):
    dataset = args.dataset
    wave_dir = dataset2wavpath[dataset]


    # Predict
    sample_uids = uids[index]
    if upaths:
        sample_upaths = upaths[index]
    else:
        sample_upaths = sample_uids + ".wav"

    wave_file = os.path.join(wave_dir, sample_upaths)
    f0, sp, ap, fs = extract_sp.extract_world_features(wave_file)
    frame_len = len(f0)
    pred = pred.cpu()
    mcep = pred[:frame_len]
    mecp_get = mecp_get[:frame_len]
    print("the shape of mcep is:", mcep.shape)
    sp_pred = extract_mcep.mgc2SP(mcep)
    sp_get = extract_mcep.mgc2SP(mecp_get)
    print("the shape of sp_pred is:", sp_pred.shape)
    print("the shape of sp is:", sp.shape)
    assert sp.shape == sp_pred.shape

    y_gt = extract_sp.world_synthesis(f0, sp, ap, fs)
    y_pred = extract_sp.world_synthesis(f0, sp_pred, ap, fs)
    y_get = extract_sp.world_synthesis(f0, sp_get, ap, fs)
    # save gt
    gt_file = os.path.join(output_dir, "{}_gt.wav".format(sample_uids))
    os.system("cp {} {}".format(wave_file, gt_file))
    # save WORLD synthesis gt
    world_gt_file = os.path.join(output_dir, "{}_gt_world.wav".format(sample_uids))
    save_audio(world_gt_file, y_gt, fs)
    
    world_gt_file = os.path.join(output_dir, "{}_get_world.wav".format(sample_uids))
    save_audio(world_gt_file, y_get, fs)
    # save WORLD synthesis pred
    world_pred_file = os.path.join(
        output_dir,
        "{}_pred_world.wav".format(sample_uids)
    )
    save_audio(world_pred_file, y_pred, fs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        default='../preprocess',
        help="dir containing test data",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default='outputs/v1_MCEP',
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
        default="Opencpop",
        help="the name of the testing dataset",
    )
    opt = parser.parse_args()
    
    data_path = opt.indir
    
    test_dataset = SingVoice(data_path, opt.dataset, "test") #for debug
    
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    if opt.dataset == "M4Singer":
        uids,upaths = get_uids(opt.dataset, "test")
    else:
        uids = get_uids(opt.dataset, "test") #for debug
        upaths = None
    
    config = OmegaConf.load("configs/infer/v1_MCEP_infer.yaml")
    model = instantiate_from_config(config.model)
    
    model.load_state_dict(torch.load("logs/2023-05-09T12-42-39_v1_MCEP/checkpoints/last.ckpt")["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    
    with torch.no_grad():
        with model.ema_scope():
            for i, cases in enumerate(tqdm(test_dataloader)):
                
                mask, whisper, f0, mecp =  cases["mask"], cases["whisper"], cases["f0"].long(), cases["MCEP"][0].to(device)
                # encode masked image and concat downsampled mask
                mask = mask.to(device).bool()
                if not config.model.params.identity:
                    mask_post = Resize((mask.shape[1]//4,1))(mask) #downsample mask [B,T//4,1]
                else:
                    mask_post = mask
                
                print("the shape of mask_post is:", mask_post.shape)
                c = {"whisper": whisper.to(device), "f0": f0.to(device), "mask": mask_post.to(device)}
                
                
                if not config.model.params.identity:
                    shape = (1, 3, 250, 10)
                else:
                    shape = (1, 1, 1000, 40)
                
                #sampling
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=shape[0],
                                                 shape=shape[1:],
                                                 verbose=False, eta=1)
                
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                #get the predicted mcep
                pred_mcep = torch.mean(x_samples_ddim[0],dim = 0) #take the mean of channels
                #turn the predicted mcep to sp
                criterion = nn.MSELoss(reduction='none')
                loss = criterion(pred_mcep,mecp)
                loss = torch.sum(loss * mask[0]) / torch.sum(mask[0])
                
                print("the MSE loss is:", loss)
                
                criterion2 = nn.L1Loss(reduction='none')
                loss2 = criterion2(pred_mcep,mecp)
                loss2 = torch.sum(loss2 * mask[0]) / torch.sum(mask[0])
                print("the L1 loss is:", loss2)
                
                save_pred_audios(pred_mcep,mecp.cpu(), opt, index=i, uids=uids,upaths = upaths, output_dir=opt.outdir)
                if i == 5:
                    break






