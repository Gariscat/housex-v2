from config import *
from dataset import MainstageDataset, create_splits
from model import MainstageModel
from torch.utils.data import DataLoader, random_split
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from easydict import EasyDict as edict
from copy import deepcopy
from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint
import os
import json

torch_rng = torch.Generator().manual_seed(42)
torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--extractor_name', type=str, default='resnet18')
    parser.add_argument('--transformer_num_layers', type=int, default=1)
    parser.add_argument('--loss_weight', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=3)
    parser.add_argument('--data_mode', type=str, default='full')
    parser.add_argument('--project', type=str, default='Mainstage-v2-dataset')
    parser.add_argument('--ckpt_dir', type=str, default='/root/autodl-tmp/checkpoints')
    parser.add_argument('--comment_on_save', type=str, default='')
    
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    model_config = edict({
        'extractor_name': args.extractor_name,
        'transformer_num_layers': args.transformer_num_layers,
        'loss_weight': args.loss_weight,
        'learning_rate': args.learning_rate,
        'd_model': args.d_model,
        'n_head': args.n_head,
    })
    
    model = MainstageModel(model_config)
    wb_config = deepcopy(model_config)
    wb_config.loss_weight = 'weighted' if wb_config is not None else None
    wb_config['comment'] = 'create-dataset'
    wb_config['batch_size'] = 4
    wb_config['data_mode'] = args.data_mode
    
    
    train_set = torch.load('/root/train_set.pth')
    val_set = torch.load('/root/test_set.pth')
    
    ### train_set, val_set = random_split(dataset, [0.8, 0.2], generator=torch_rng)
    
    class_cnt = sum([y for _, y in train_set])
    for genre, score in zip(ALL_GENRES, class_cnt.numpy().tolist()):
        print(genre, score)
    lw = 1 / class_cnt
    lw /= lw.sum()
    lw = torch.tensor(lw, dtype=torch.float32)

    train_loader = DataLoader(train_set, batch_size=wb_config['batch_size'], shuffle=True, generator=torch_rng)
    val_loader = DataLoader(val_set, batch_size=wb_config['batch_size'], shuffle=False, generator=torch_rng)
    
    wandb_logger = WandbLogger(
        project=args.project,
        config=wb_config,
        save_dir='/root'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',    # The metric to monitor (validation accuracy in this case)
        mode='max',                # Save the checkpoint with the maximum accuracy
        save_top_k=1,              # Save only the best checkpoint
        dirpath=args.ckpt_dir,    # Directory where the checkpoints will be saved
        filename=f'{args.extractor_name}-{args.transformer_num_layers}-{args.n_head}' # Filename for the best checkpoint
    )
    
    trainer = L.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=10,
        logger=wandb_logger,
        log_every_n_steps=1,
        val_check_interval=0.25,
        # enable_checkpointing=False,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    model.load_from_checkpoint(checkpoint_callback.best_model_path)
    with open(os.path.join(args.ckpt_dir, f'{args.extractor_name}-{args.transformer_num_layers}-{args.n_head}-{args.comment_on_save}.json'), 'w') as f:
        ret = {}
        ret['train_res'] = model.train_metric_results
        ret['val_res'] = model.val_metric_results
        json.dump(ret, f)
        
        print('Results saved to', f.name)
