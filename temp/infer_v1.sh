#!/bin/sh

#SBATCH -J diffsvc
#SBATCH -o /mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/latent_diffusion/infer_log/V1-%j.log
#SBATCH -e /mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/latent_diffusion/infer_log/V1-%j.err
#SBATCH -N 1 -n 1 -c 4 -p p-RTX2080 --gres=gpu:1
#SBATCH --time=24:00:00



source activate ldm

cd /mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/latent_diffusion/


python infer_v1.py
