#!/bin/sh

#SBATCH -J diffsvc
#SBATCH -o /mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/latent_diffusion/infer_log/diffsvc-%j.log
#SBATCH -e /mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/latent_diffusion/infer_log/diffsvc-%j.err
#SBATCH -N 1 -n 1 -c 4 -p p-RTX2080 --gres=gpu:1
#SBATCH --time=24:00:00



source activate ldm

cd /mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/latent_diffusion/


python infer.py --outdir outputs/diffsvc --steps 50 --dataset M4Singer --checkpoint logs/diffsvc/checkpoints/best.ckpt --config configs/infer/diffsvc.yaml --ratio 0.5
