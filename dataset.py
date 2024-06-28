import os
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import json
from detect import AUDIO_DIR
import matplotlib.pyplot as plt
from config import *
from tqdm import tqdm


def get_power_mel_spectrogram(y: np.ndarray, sr: int, eps: float=1e-5, debug: bool=False):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    
    # print(np.min(log_S), np.max(log_S))
    assert np.min(log_S) >= -80.0 - eps and np.max(log_S) <= 0.0 + eps
    
    if debug:
        plt.imshow(-log_S/80)
        plt.colorbar()
        plt.show()
        plt.close()
        
    return -log_S / 80.0


class HouseXDataset(Dataset):
    def __init__(self,
        drop_detection_path: str,
        genre_annotation_path: str,
    ):
        super(HouseXDataset, self).__init__()
        with open(drop_detection_path, "r") as f:
            self.detected_drops = json.load(f)
        with open(genre_annotation_path, "r") as f:
            self.genre_annotations = json.load(f)
            
        self._data = []
            
        for track_info in tqdm(self.genre_annotations, desc="Loading tracks"):
            # assert len(track_info['annotations']) == 1
            track_name = os.path.basename(track_info['data']['audio'])
            track_absolute_path = os.path.join(AUDIO_DIR, track_name)
            
            genre_soft_label = np.zeros(len(ALL_GENRES))
            annotation_results = track_info['annotations'][0]['result']
            for genre_info in annotation_results:
                genre_id = ALL_GENRES.index(genre_info['from_name'])
                genre_soft_label[genre_id] = genre_info['value']["number"]
                
            assert genre_soft_label.sum() == 1.0
            
            y, sr = librosa.load(track_absolute_path, mono=True)
            
            drop_sections = None
            for drop in self.detected_drops:
                if drop["audio_path"] == track_absolute_path:
                    drop_sections = drop["drop_sections"]
                    break
                
            if drop_sections is None:
                continue
            
            # print("track:", track_name)
            for drop_st, drop_ed in drop_sections:
                drop_st_sample = librosa.time_to_samples(drop_st, sr=sr)
                drop_ed_sample = librosa.time_to_samples(drop_ed, sr=sr)
                num_sample_per_clip = int(NUM_SECONDS_PER_CLIP * sr)
                
                # print("  drop_loop_length:", drop_ed_sample - drop_st_sample)
                # print("  clip_length:", num_sample_per_clip)
                
                for _ in range(NUM_CLIP_PER_DROPLOOP):
                    clip_st = np.random.randint(drop_st_sample, drop_ed_sample - num_sample_per_clip)
                    clip_ed = clip_st + num_sample_per_clip
                    clip = y[clip_st:clip_ed]
                    
                    melspec = get_power_mel_spectrogram(clip, sr)
                    self._data += [{
                        "spec": torch.from_numpy(melspec),
                        "label": torch.from_numpy(genre_soft_label)
                    }]
                    
        print("total clips:", len(self._data))
            
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx: int):
        return self._data[idx]
    
    
    
if __name__ == "__main__":
    drop_detection_path = "detected_drops.json"
    genre_annotation_path = "/Users/ca7ax/housex-v2/project-7-at-2024-06-28-07-55-4149f847.json"
    dataset = HouseXDataset(drop_detection_path, genre_annotation_path)
    