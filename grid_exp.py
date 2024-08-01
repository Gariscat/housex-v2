import os
import subprocess
from itertools import product

EXTRACTORS = ('resnet18', 'resnet50', 'resnet152',)
T_NUM_LAYERS = (1, 2, 4,)
N_HEADS = (1, 2, 4,)

if __name__ == '__main__':
    for e, t, n in product(EXTRACTORS, T_NUM_LAYERS, N_HEADS):
        subprocess.call(f'python train.py \
                    --extractor_name {e} \
                    --transformer_num_layer {t} \
                    --n_head {n}', \
                    shell=True
                )