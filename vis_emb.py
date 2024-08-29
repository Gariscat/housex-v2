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

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from mpl_toolkits.mplot3d import Axes3D


def visualize_embeddings_pca(embeddings, labels, class_names, n_components=2, save_path='pca_plot.png'):
    """
    Visualize the latent embeddings using PCA and save the plot to the specified path.

    Args:
        embeddings (torch.Tensor): The latent embeddings with shape (seq_len, d_embedding).
        labels (torch.Tensor): The labels with shape (seq_len, class_cnt) or (seq_len,).
        class_names (list of str): List of class names for the legend.
        n_components (int): Number of components for PCA (2 or 3 for 2D or 3D visualization).
        save_path (str): File path to save the PCA plot.

    Returns:
        None. Saves the PCA plot to the specified file.
    """
    
    # Ensure embeddings and labels are on CPU and convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    if labels.ndim > 1:
        labels_np = labels.argmax(dim=1).cpu().numpy()
    else:
        labels_np = labels.cpu().numpy()

    # PCA
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings_np)

    # Plotting PCA
    plt.figure(figsize=(7, 5))
    if n_components == 2:
        scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=labels_np, cmap='viridis', alpha=0.7)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
    elif n_components == 3:
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], embeddings_pca[:, 2], c=labels_np, cmap='viridis', alpha=0.7)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')

    plt.title('PCA of Embeddings')
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Classes")
    plt.savefig(str(n_components)+'-dim-'+save_path)
    plt.close()


def visualize_embeddings_tsne(embeddings, labels, class_names, n_components=2, save_path='tsne_plot.png'):
    """
    Visualize the latent embeddings using t-SNE and save the plot to the specified path.

    Args:
        embeddings (torch.Tensor): The latent embeddings with shape (seq_len, d_embedding).
        labels (torch.Tensor): The labels with shape (seq_len, class_cnt) or (seq_len,).
        class_names (list of str): List of class names for the legend.
        n_components (int): Number of components for t-SNE (2 or 3 for 2D or 3D visualization).
        save_path (str): File path to save the t-SNE plot.

    Returns:
        None. Saves the t-SNE plot to the specified file.
    """

    # Ensure embeddings and labels are on CPU and convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    if labels.ndim > 1:
        labels_np = labels.argmax(dim=1).cpu().numpy()
    else:
        labels_np = labels.cpu().numpy()

    # t-SNE
    tsne = TSNE(n_components=n_components, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings_np)

    # Plotting t-SNE
    plt.figure(figsize=(7, 5))
    if n_components == 2:
        scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels_np, cmap='viridis', alpha=0.7)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
    elif n_components == 3:
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], embeddings_tsne[:, 2], c=labels_np, cmap='viridis', alpha=0.7)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_zlabel('t-SNE Component 3')

    plt.title('t-SNE of Embeddings')
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Classes")
    plt.savefig(str(n_components)+'-dim-'+save_path)
    plt.close()


def visualize_embeddings_umap(embeddings, labels, class_names, n_components=2, save_path='umap_plot.png'):
    """
    Visualize the latent embeddings using UMAP and save the plot to the specified path.

    Args:
        embeddings (torch.Tensor): The latent embeddings with shape (seq_len, d_embedding).
        labels (torch.Tensor): The labels with shape (seq_len, class_cnt) or (seq_len,).
        class_names (list of str): List of class names for the legend.
        n_components (int): Number of components for UMAP (2 or 3 for 2D or 3D visualization).
        save_path (str): File path to save the UMAP plot.

    Returns:
        None. Saves the UMAP plot to the specified file.
    """

    # Ensure embeddings and labels are on CPU and convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    if labels.ndim > 1:
        labels_np = labels.argmax(dim=1).cpu().numpy()
    else:
        labels_np = labels.cpu().numpy()

    # UMAP
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embeddings_umap = reducer.fit_transform(embeddings_np)

    # Plotting UMAP
    plt.figure(figsize=(7, 5))
    if n_components == 2:
        scatter = plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=labels_np, cmap='viridis', alpha=0.7)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
    elif n_components == 3:
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], embeddings_umap[:, 2], c=labels_np, cmap='viridis', alpha=0.7)
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_zlabel('UMAP Component 3')

    plt.title('UMAP of Embeddings')
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Classes")
    plt.savefig(str(n_components)+'-dim-'+save_path)
    plt.close()
    
'''
    
def visualize_embeddings_pca(embeddings, labels, class_names, n_components=2, save_path='pca_plot.png'):
    """
    Visualize the latent embeddings using PCA and save the plot to the specified path.

    Args:
        embeddings (torch.Tensor): The latent embeddings with shape (seq_len, d_embedding).
        labels (torch.Tensor): The labels with shape (seq_len, class_cnt) or (seq_len,).
        class_names (list of str): List of class names for the legend.
        n_components (int): Number of components for PCA (default is 2 for 2D visualization).
        save_path (str): File path to save the PCA plot.

    Returns:
        None. Saves the PCA plot to the specified file.
    """

    # Ensure embeddings and labels are on CPU and convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    if labels.ndim > 1:
        labels_np = labels.argmax(dim=1).cpu().numpy()
    else:
        labels_np = labels.cpu().numpy()

    # PCA
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings_np)

    # Plotting PCA
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=labels_np, cmap='viridis', alpha=0.7)
    plt.title('PCA of Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # Update legend with class names
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Classes")
    plt.savefig(str(n_components)+'-'+save_path)
    plt.close()


def visualize_embeddings_tsne(embeddings, labels, class_names, n_components=2, save_path='tsne_plot.png'):
    """
    Visualize the latent embeddings using t-SNE and save the plot to the specified path.

    Args:
        embeddings (torch.Tensor): The latent embeddings with shape (seq_len, d_embedding).
        labels (torch.Tensor): The labels with shape (seq_len, class_cnt) or (seq_len,).
        class_names (list of str): List of class names for the legend.
        n_components (int): Number of components for t-SNE (default is 2 for 2D visualization).
        save_path (str): File path to save the t-SNE plot.

    Returns:
        None. Saves the t-SNE plot to the specified file.
    """

    # Ensure embeddings and labels are on CPU and convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    if labels.ndim > 1:
        labels_np = labels.argmax(dim=1).cpu().numpy()
    else:
        labels_np = labels.cpu().numpy()

    # t-SNE
    tsne = TSNE(n_components=n_components, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings_np)

    # Plotting t-SNE
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels_np, cmap='viridis', alpha=0.7)
    plt.title('t-SNE of Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    # Update legend with class names
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Classes")
    plt.savefig(str(n_components)+'-'+save_path)
    plt.close()


def visualize_embeddings_umap(embeddings, labels, class_names, n_components=2, save_path='umap_plot.png'):
    """
    Visualize the latent embeddings using UMAP and save the plot to the specified path.

    Args:
        embeddings (torch.Tensor): The latent embeddings with shape (seq_len, d_embedding).
        labels (torch.Tensor): The labels with shape (seq_len, class_cnt) or (seq_len,).
        class_names (list of str): List of class names for the legend.
        n_components (int): Number of components for UMAP (default is 2 for 2D visualization).
        save_path (str): File path to save the UMAP plot.

    Returns:
        None. Saves the UMAP plot to the specified file.
    """

    # Ensure embeddings and labels are on CPU and convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    if labels.ndim > 1:
        labels_np = labels.argmax(dim=1).cpu().numpy()
    else:
        labels_np = labels.cpu().numpy()

    # UMAP
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embeddings_umap = reducer.fit_transform(embeddings_np)

    # Plotting UMAP
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=labels_np, cmap='viridis', alpha=0.7)
    plt.title('UMAP of Embeddings')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    # Update legend with class names
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Classes")
    plt.savefig(str(n_components)+'-'+save_path)
    plt.close()
'''


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
    parser.add_argument('--gpu_id', type=int, default=4)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--ckpt_path', type=str, default='/home/xinyu.li/checkpoints/full-True/resnet152-1-6.ckpt')
    
    parser.add_argument('--force_run', default=False, action='store_true')
    parser.add_argument('--set', type=str, default='val')
    parser.add_argument('--n_comp', type=int, default=3)
    
    args = parser.parse_args()
    
    if args.force_run:
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
        if args.set == 'val':
            trainer.validate(model=model, dataloaders=val_loader)
        else:
            trainer.validate(model=model, dataloaders=train_loader)
    
    # Now the embeddings are saved
    
    tensors = torch.load('/home/xinyu.li/my_emb_lab.pth')
    emb, label = tensors['emb'], tensors['label']
    
    print(emb.shape, label.shape)
    
    # Random example data
    seq_len, d_embedding = emb.shape
    class_cnt = len(ALL_GENRES)
    
    """# Example embeddings and labels
    embeddings = torch.randn(seq_len, d_embedding)
    labels = torch.randint(0, class_cnt, (seq_len,))  # Random integer labels"""

    # Visualize using PCA
    visualize_embeddings_pca(emb, label, class_names=ALL_GENRES, n_components=args.n_comp)
    
    # Visualize using t-SNE
    visualize_embeddings_tsne(emb, label, class_names=ALL_GENRES, n_components=args.n_comp)
    
    # Visualize using UMAP
    visualize_embeddings_umap(emb, label, class_names=ALL_GENRES, n_components=args.n_comp)