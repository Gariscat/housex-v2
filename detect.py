"""This file contains functions to detect drop in an EDM track."""

import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import json
from config import AUDIO_DIR, FRAME_LENGTH
from utils import find_drop, max_smooth, avg_smooth
import subprocess
    
    
if __name__ == "__main__":
    # ret = find_drop("/Users/ca7ax/Downloads/audio/Nothing To Hide - PeTE _ Trevor Omoto _ Peter Pentsak.ogg", True)
    # print(ret)
    # find_drop("/Users/ca7ax/Downloads/Martin Garrix - Now That I've Found You (feat. John & Michel) [Official Video].mp3")
    # find_drop("/Users/ca7ax/Downloads/Martin Garrix & Third Party - Lions In The Wild [Official Video].mp3")
    drop_annotations = []
    
    detected = 0
    
    for audio_name in tqdm(os.listdir(AUDIO_DIR)):
        audio_path = os.path.join(AUDIO_DIR, audio_name)
        """annotation = find_drop(audio_path)"""
        subprocess.call(f'python utils.py \
                    --audio_path {audio_path}', \
                    shell=True
                )
        with open('tmp.json', 'r') as f:
            annotation = json.load(f)
        drop_annotations += [annotation]
        detected += int(annotation['drop_sections'] != [])
        
    print(f"Detection rate: {detected / len(drop_annotations)}")
        
    with open('annotations/detected_drops.json', 'w') as f:
        json.dump(drop_annotations, f)