import os
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import json
# from detect import AUDIO_DIR
import matplotlib.pyplot as plt
from config import *
from tqdm import tqdm
from skimage.transform import resize
from utils import read_audio_st_ed
from typing import List, Tuple
import random

def get_power_mel_spectrogram(y: np.ndarray, sr: int, eps: float=1e-5, debug: bool=False):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224)
    log_S = librosa.power_to_db(S, ref=np.max)
    
    # print(np.min(log_S), np.max(log_S))
    assert np.min(log_S) >= -80.0 - eps and np.max(log_S) <= 0.0 + eps
    
    if debug:
        plt.imshow(-log_S/80)
        plt.colorbar()
        plt.show()
        plt.close()
        
    # return -log_S / 80.0
    return np.expand_dims(-log_S / 80.0, axis=0).astype(np.float32)

def get_chromagrams(y: np.ndarray, sr: int, intervals: str='ji5', debug: bool=False):
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vq = librosa.feature.chroma_vqt(y=y, sr=sr, intervals=intervals)
    
    n = chroma_cq.shape[1]
    
    chroma_cq = resize(chroma_cq, (224, n))
    chroma_vq = resize(chroma_vq, (224, n))
    ret = np.stack([chroma_cq, chroma_vq])
    return ret.astype(np.float32)
    """fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time',
                            ax=ax[0])
    ax[0].set(ylabel='chroma_cqt')
    ax[0].label_outer()
    img = librosa.display.specshow(chroma_vq, y_axis='chroma_fjs', x_axis='time',
                                ax=ax[1],
                                intervals='pythagorean')
    ax[1].set(ylabel='chroma_vqt')
    fig.colorbar(img, ax=ax)
    plt.show()
    plt.close()
    print(chroma_cq.shape, chroma_vq.shape)
    print(chroma_cq.min(), chroma_cq.max())
    print(chroma_vq.min(), chroma_vq.max())"""
    
def get_gram(clip: np.ndarray, sr: int, use_chroma: bool=False):
    melspec = get_power_mel_spectrogram(clip, sr)
                    
    if not use_chroma:
        gram = torch.from_numpy(melspec).repeat(3, 1, 1)
    else:
        chroma = get_chromagrams(clip, sr)
                        
        melspec = torch.from_numpy(melspec)
        chroma = torch.from_numpy(chroma)
        gram = torch.cat([melspec, chroma], dim=0)
    
    return gram

def process_audio_dir(audio_dir: str) -> List:
    drop_detection_path = [x for x in os.listdir(audio_dir) if x.endswith('.json') and 'detect' in x][0]
    genre_annotation_path = [x for x in os.listdir(audio_dir) if x.endswith('.json') and 'refined' in x][0]
    with open(os.path.join(audio_dir, drop_detection_path), "r") as f:
        detected_drops = json.load(f)
    with open(os.path.join(audio_dir, genre_annotation_path), "r") as f:
        genre_annotations = json.load(f)
        
    ret = [] # (path, soft_label, drop_sections)
    
    for track_info in tqdm(genre_annotations, desc="Reading tracks"):
        # assert len(track_info['annotations']) == 1
            ### PROCESS GENRE LABELS
        track_name = os.path.basename(track_info['data']['audio'])
        track_absolute_path = os.path.join(audio_dir, track_name)
            
        genre_soft_label = np.zeros(len(ALL_GENRES))
        annotation_results = track_info['annotations'][0]['result']
        for genre_info in annotation_results:
            genre_id = ALL_GENRES.index(genre_info['from_name'])
            genre_soft_label[genre_id] = genre_info['value']["number"]
            
        # genre_soft_label = torch.from_numpy(genre_soft_label).float()
        try:
            assert genre_soft_label.sum().item() == 1.0
        except:
            print(track_name, genre_soft_label)
            genre_soft_label /= genre_soft_label.sum()
            
            ### PROCESS AUDIO DATA
            ### y, sr = librosa.load(track_absolute_path, mono=True)
            ### y, sr = read_audio(track_absolute_path)
            
        drop_sections = None
        for drop in detected_drops:
            if os.path.basename(drop["audio_path"]) == os.path.basename(track_absolute_path):
                drop_sections = drop["drop_sections"]
                break
                
        if drop_sections is None:
            continue
        
        ret.append((track_absolute_path, genre_soft_label.tolist(), drop_sections))
        
    return ret

def create_splits(audio_dirs: List[str], split_ratio: List[float], rng_seed: int=42) -> List[List]:
    data_list = []
    for audio_dir in audio_dirs:
        data_list += process_audio_dir(audio_dir)
        
    random.seed(rng_seed)
    random.shuffle(data_list)
    
    assert sum(split_ratio) == 1.0
    
    boundaries = (np.cumsum([0]+split_ratio) * len(data_list)).astype(int)
    
    splits = []
    for i in range(boundaries.shape[0] - 1):
        cur_split = data_list[boundaries[i]:boundaries[i+1]]
        splits.append(cur_split)
        
    return splits
        
class HouseXDataset(Dataset):
    def __init__(self,
        data_list: List[Tuple],
        # use_mel_spectrogram: bool=True,
        use_chroma: bool=True,
    ):
        super(HouseXDataset, self).__init__()
        
        self.track_names = []
        self._data = []
        
        for track_absolute_path, genre_soft_label, drop_sections in tqdm(data_list, desc="Creating dataset"):
            # assert len(track_info['annotations']) == 1
            
            self.track_names.append(os.path.basename(track_absolute_path))
            
            genre_soft_label = torch.tensor(genre_soft_label).float()
            
            for drop_st, drop_ed in drop_sections:
                y_cur, sr = read_audio_st_ed(track_absolute_path, drop_st, drop_ed)
                drop_st_sample = librosa.time_to_samples(drop_st, sr=sr)
                drop_ed_sample = librosa.time_to_samples(drop_ed, sr=sr)
                num_sample_per_clip = int(NUM_SECONDS_PER_CLIP * sr)
                
                ### print("  drop_loop_length:", drop_ed_sample - drop_st_sample)
                ### print("  clip_length:", num_sample_per_clip)
                
                for _ in range(NUM_CLIP_PER_DROPLOOP):
                    clip_st = np.random.randint(0, drop_ed_sample - drop_st_sample - num_sample_per_clip)
                    clip_ed = clip_st + num_sample_per_clip
                    clip = y_cur[clip_st:clip_ed]
                    
                    gram = get_gram(clip, sr, use_chroma)
                    
                    self._data.append((gram, genre_soft_label))
                    
        print("total clips:", len(self._data))
            
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx: int):
        return self._data[idx]
    
    
    
if __name__ == "__main__":
    pass
