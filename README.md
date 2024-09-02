# 🔥 Mainstage EDM Sub-Genre Benchmark 🔥
An extended classification benchmark for mainstream dance music in the style of house, covering progressive house, future house, bass house, tech house, deep house, bigroom, future rave and slap house.

## Collection

Our dataset contains 1035 tracks. You can download it from [Google Drive](https://drive.google.com/drive/folders/12VNfriD5d6aUGN5w-LVKxT0Gsqe7RpEL?usp=sharing). We split them into 3 parts due to the file upload limit in ```Label-Studio``` and that the data is labeled by multiple experts. These 3 parts are **not** 3 folds of the dataset in the sense of train/val/test. The splits are created after loading all the tracks from the folders.

## Annotation

Annotation is done using ```Label-Studio``` (MANY thanks to the developers!). We use soft labeling such that the probabilities of each sub-genre should sum up to 1.

## Drop Detection

Modify the directories in ```detect.py``` to the 3 folders of our data (or your own data with the same structure), Run ```python detect.py``` to detect the drops of tracks using rule-based algorithm by volume thresholds. The detected drops are stored as ```.json``` files in the folders respectively.

## Dataset Generation

Again, modify the directories in the ```main``` function of ```dataset.py``` to the corresponding folders. Then, run ```python dataset.py``` to generate the training set and the validation set (which is also the test set in our context). You can specify the ```--use_chroma``` and ```--mode``` parameter to determine whether to include chromagrams in the data representation and which type of label (soft/hard) to use.

## Training

Run ```python train.py```. Again, use the above mentioned 2 parameters to specify the dataset. Also, you can control other parameters like the network architecture and which GPU to use (Please refer to ```train.py``` for details). After training, you can run ```python vis_emb.py --force_run``` to visualize the embeddings with dimension reduction techniques like PCA, t-SNE and UMAP. Make sure that the checkpoint you load is trained from the dataset that matches the parameters in ```vis_emb.py```. Checkpoints trained on dataset without chromagrams would give random scatter points on dataset with chromagrams :|

## Deployment

The demo of stage visuals controlled by our classification model is under construction. Coming soon in a week or two :)

## Checking existing annotations
We also prepared a simple UI to check existing annotations (.csv). Run ```utils.py``` to view annotations of audio files in a folder. Before you run, please make sure you have modified ```anno_path``` and ```audio_dir``` in the .py file to your corresponding paths.
