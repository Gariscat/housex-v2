from utils import find_drop
from config import *
from dataset import MainstageDataset
from model import MainstageModel
from torch.utils.data import DataLoader, random_split
import torch
import lightning as L
from easydict import EasyDict as edict
from copy import deepcopy
from argparse import ArgumentParser
import os
import json
from dataset import get_gram, get_chromagrams, get_power_mel_spectrogram
from tqdm import tqdm
import librosa
import numpy as np
from config import ALL_GENRES, NUM_SECONDS_PER_CLIP
from pprint import pprint
from datetime import timedelta

torch_rng = torch.Generator().manual_seed(42)
torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--extractor_name', type=str, default='resnet152')
    parser.add_argument('--transformer_num_layers', type=int, default=1)
    parser.add_argument('--loss_weight', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--use_chroma', default=False, action='store_true')
    parser.add_argument('--mode', type=str, default='full')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='/root/resnet152-1-6.ckpt')
    parser.add_argument('--track_path', type=str)
    
    args = parser.parse_args()
    
    model_config = edict({
        'extractor_name': args.extractor_name,
        'transformer_num_layers': args.transformer_num_layers,
        'loss_weight': args.loss_weight,
        'learning_rate': args.learning_rate,
        'd_model': args.d_model,
        'n_head': args.n_head,
    })
    model = MainstageModel(model_config)
    model = MainstageModel.load_from_checkpoint(args.ckpt_path, model_config=model_config)
    
    model.eval()
    model.to(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 else 'cpu')
    model.config.output_embedding = False
    
    y_track, sr = librosa.load(args.track_path, mono=True)
    drop_sections = find_drop(args.track_path, write_to_tmp=False)['drop_sections']
    num_sample_per_clip = int(NUM_SECONDS_PER_CLIP * sr)
    print(drop_sections)
    
    for drop_section in drop_sections:
        drop_st_sample = librosa.time_to_samples(drop_section[0], sr=sr)
        drop_ed_sample = librosa.time_to_samples(drop_section[1], sr=sr)
        
        for cur_sample in np.linspace(drop_st_sample, drop_ed_sample - num_sample_per_clip, 4, dtype=int):
            clip_st_sample = cur_sample
            clip_ed_sample = clip_st_sample + num_sample_per_clip
            y_clip = y_track[clip_st_sample:clip_ed_sample]
            feat = get_gram(y_clip, sr, args.use_chroma)
            feat = feat.unsqueeze(0).to(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 else 'cpu')
            
            logits = model(feat).detach().cpu()
            probs = torch.softmax(logits, dim=-1).flatten().numpy()
            ret = {
                "Audio path:": args.track_path,
                "st_sec:": librosa.samples_to_time(clip_st_sample, sr=sr),
                "ed_sec": librosa.samples_to_time(clip_ed_sample, sr=sr),
                "Probs:": probs.tolist(),
                "Prediction:": ALL_GENRES[probs.argmax()]
            }
            pprint(ret)