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
from utils import sharpen_label, compute_metrics
from tqdm import tqdm

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
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='/home/xinyu.li/checkpoints/full-False/resnet152-1-6.ckpt')
    
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
    
    train_set = torch.load(f'/home/xinyu.li/train_set_{args.mode}_{args.use_chroma}.pth')
    val_set = torch.load(f'/home/xinyu.li/test_set_{args.mode}_{args.use_chroma}.pth')
    if args.debug:
        train_set = train_set[:100]
        val_set = val_set[:20]
    
    ### train_set, val_set = random_split(dataset, [0.8, 0.2], generator=torch_rng)
    
    class_cnt = sum([y for _, y in train_set])
    for genre, score in zip(ALL_GENRES, class_cnt.numpy().tolist()):
        print(genre, score)
    lw = 1 / class_cnt
    lw /= lw.sum()
    lw = torch.tensor(lw, dtype=torch.float32)

    # model = MainstageModel.__init__(model_config).load_from_checkpoint(checkpoint_callback.best_model_path)
    model = MainstageModel.load_from_checkpoint(args.ckpt_path, model_config=model_config)
    print("Best ckpt reloaded.")
    model.eval()
    
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, generator=torch_rng)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, generator=torch_rng)
    
    trainer = L.Trainer(
        max_epochs=1 if args.debug else 5,
        log_every_n_steps=1,
        val_check_interval=0.5,
        devices=[args.gpu_id,],
        accelerator="gpu"
        # enable_checkpointing=False,
    )
    
    model.config.output_embedding = True
    trainer.validate(model=model, dataloaders=val_loader)
    
    # Now the embeddings are saved
    
    tensors = torch.load('/home/xinyu.li/emb_lab.pth')
    emb, label = tensors['emb'], tensors['label']