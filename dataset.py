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
from utils import read_audio_st_ed, sharpen_label
from typing import List, Tuple
import random
import soundfile as sf
from argparse import ArgumentParser

def get_power_mel_spectrogram(y: np.ndarray, sr: int, eps: float=1e-5, debug: bool=False):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
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
    
    chroma_cq = resize(chroma_cq, (N_MELS, n))
    chroma_vq = resize(chroma_vq, (N_MELS, n))
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

def process_audio_dir(audio_dir: str, mode: str='full') -> List:
    drop_detection_path = [x for x in os.listdir(audio_dir) if x.endswith('.json') and 'detect' in x][0]
    genre_annotation_path = [x for x in os.listdir(audio_dir) if x.endswith('.json') and 'partition' in x][0]
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
            
        genre_soft_label = torch.from_numpy(genre_soft_label).float()
        try:
            assert genre_soft_label.sum().item() == 1.0
        except:
            print(track_name, genre_soft_label)
            genre_soft_label /= genre_soft_label.sum()
            
        if mode == 'sharpen': # sharpen soft label to hard (0/1) label
            genre_soft_label = sharpen_label(genre_soft_label)
        elif mode == 'clean': # discard soft labels
            if (genre_soft_label==1).any().item() is False:
                continue
            
            
        drop_sections = None
        for drop in detected_drops:
            if os.path.basename(drop["audio_path"]) == os.path.basename(track_absolute_path):
                drop_sections = drop["drop_sections"]
                break
                
        if drop_sections is None: # drop not detected by rule-based algo
            continue
        
        ret.append((track_absolute_path, genre_soft_label, drop_sections))
        
    return ret

def create_splits(audio_dirs: List[str], split_ratio: List[float], rng_seed: int=42, mode: str='full') -> List[List]:
    data_list = []
    for audio_dir in audio_dirs:
        data_list += process_audio_dir(audio_dir, mode=mode)
        
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
        use_chroma: bool=False,
        audio_standalone_dir: str=None,
    ):
        super(HouseXDataset, self).__init__()
        
        self.track_names = []
        self._data = []
        if audio_standalone_dir is not None:
            import shutil
            shutil.rmtree(audio_standalone_dir, ignore_errors=True)
            os.makedirs(audio_standalone_dir, exist_ok=False)
            self.clip_info = []
        
        for track_absolute_path, genre_soft_label, drop_sections in tqdm(data_list, desc="Creating dataset"):
            # assert len(track_info['annotations']) == 1
            
            self.track_names.append(os.path.basename(track_absolute_path))
            
            # genre_soft_label = torch.tensor(genre_soft_label).float()
            
            for drop_st, drop_ed in drop_sections:
                y_cur, sr = read_audio_st_ed(track_absolute_path, drop_st, drop_ed)
                drop_st_sample = librosa.time_to_samples(drop_st, sr=sr)
                drop_ed_sample = librosa.time_to_samples(drop_ed, sr=sr)
                num_sample_per_clip = int(NUM_SECONDS_PER_CLIP * sr)
                
                ### print("  drop_loop_length:", drop_ed_sample - drop_st_sample)
                ### print("  clip_length:", num_sample_per_clip)
                
                for i in range(NUM_CLIP_PER_DROPLOOP):
                    clip_st = np.random.randint(0, drop_ed_sample - drop_st_sample - num_sample_per_clip)
                    clip_ed = clip_st + num_sample_per_clip
                    clip = y_cur[clip_st:clip_ed]
                    
                    gram = get_gram(clip, sr, use_chroma)
                    
                    self._data.append((gram, genre_soft_label))
                    
                    if audio_standalone_dir is not None:
                        standalone_path = os.path.join(audio_standalone_dir, str(i+1)+os.path.basename(track_absolute_path))
                        sf.write(standalone_path, clip, sr)
                        
                        self.clip_info.append({
                            'track_path': standalone_path,
                            'clip_start_time': librosa.samples_to_time(clip_st, sr=sr),
                            'clip_end_time': librosa.samples_to_time(clip_ed, sr=sr),
                        })
                    
        print("total clips:", len(self._data))
        if audio_standalone_dir:
            with open(os.path.join(audio_standalone_dir, 'clip_info.json'), 'w') as f:
                json.dump(self.clip_info, f)
            
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx: int):
        return self._data[idx]
    
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='full')
    args = parser.parse_args()
    
    train_test_ratio = [0.8, 0.2]
    train_split, test_split = create_splits(
        audio_dirs=['/root/part-1-5/', '/root/part-6-10/', '/root/part-new/', ],
        split_ratio=train_test_ratio,
        rng_seed=42,
        mode=args.mode,
    )
    
    train_set = HouseXDataset(data_list=train_split, use_chroma=True, audio_standalone_dir='/root/standalone_train/')
    val_set = HouseXDataset(data_list=test_split, use_chroma=True, audio_standalone_dir='/root/standalone_test/')
    torch.save(train_set, '/root/train_set.pth')
    torch.save(val_set, '/root/test_set.pth')
