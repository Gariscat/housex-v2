import os
import subprocess
from itertools import product
from argparse import ArgumentParser

# EXTRACTORS = ('resnet18', 'resnet50', 'resnet152',)
EXTRACTORS = ('resnet152', 'densenet201', 'vgg11_bn', 'resnext101_32x8d', 'vit_b_16')
# T_NUM_LAYERS = (1, 2, 4,)
T_NUM_LAYERS = (1, )
# N_HEADS = (1, 2, 4,)
N_HEADS = (6,)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--use_chroma', default=False, action='store_true')
    parser.add_argument('--mode', type=str, default='full')
    parser.add_argument('--gpu_id', type=int, default=-1)
    args = parser.parse_args()
    
    for e, t, n in product(EXTRACTORS, T_NUM_LAYERS, N_HEADS):
        subprocess.call(f'python train.py \
                    --extractor_name {e} \
                    --transformer_num_layer {t} \
                    --n_head {n} \
                    --project {"hx-v2-new-data-full-report"} \
                    --mode {args.mode} \
                    --gpu_id {args.gpu_id} \
                    {"--use_chroma" if args.use_chroma else ""}', \
                    shell=True
                )
        
    os.system("/usr/bin/shutdown")
