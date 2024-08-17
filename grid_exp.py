import os
import subprocess
from itertools import product

# EXTRACTORS = ('resnet18', 'resnet50', 'resnet152',)
EXTRACTORS = ('resnet152', 'densenet201', 'vgg11_bn', 'resnext101_32x8d', 'vit_b_16')
# T_NUM_LAYERS = (1, 2, 4,)
T_NUM_LAYERS = (1, )
# N_HEADS = (1, 2, 4,)
N_HEADS = (6,)

if __name__ == '__main__':
    for e, t, n in product(EXTRACTORS, T_NUM_LAYERS, N_HEADS):
        subprocess.call(f'python train.py \
                    --extractor_name {e} \
                    --transformer_num_layer {t} \
                    --n_head {n} \
                    --project {"hx-v2-new-data-full-report"}', \
                    shell=True
                )
        
    os.system("/usr/bin/shutdown")
