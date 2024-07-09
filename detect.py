"""This file contains functions to detect drop in an EDM track."""

import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import json
from config import AUDIO_DIR, FRAME_LENGTH


def max_smooth(a: np.ndarray, n: int) -> np.ndarray:
    """Smooths the input array by maximizing over an interval of length $n$.

    Args:
        a (np.ndarray): The input array.
        n (int): The number of elements to maximize over.

    Returns:
        np.ndarray: The smoothed array.
    """
    ret = np.zeros_like(a)
    for i in range(len(a)):
        st = max(0, i-n//2)
        ed = min(len(a), i+n//2)
        ret[i] = np.max(a[st:ed])
    return ret

def avg_smooth(a: np.ndarray, n: int) -> np.ndarray:
    """Smooths the input array by averaging over an interval of length $n$.

    Args:
        a (np.ndarray): The input array.
        n (int): The number of elements to average over.

    Returns:
        np.ndarray: The smoothed array.
    """
    return np.convolve(a, np.ones(n), 'same') / n
    

def find_drop(audio_path: str, debug: bool=False):
    """Detects the drop in an EDM track.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        sections (List[str]): pairs of (st, ed) of drops
    """
    y, sr = librosa.load(audio_path)
    ### y = y[15*sr:]  # truncate the first 15 seconds
    loudness = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH).flatten()
    # Convert loudness from rms to dB
    loudness_db = librosa.amplitude_to_db(loudness, ref=np.max)
    if debug:
        frames = range(len(loudness_db))
        t = librosa.frames_to_time(frames, sr=sr)
        # Plot the loudness curve
        plt.figure(figsize=(10, 4))
        plt.plot(t, loudness_db, color='b')
        plt.xlabel('Time (s)')
        plt.ylabel('Loudness')
        plt.title('Loudness Curve')
        plt.show()
        plt.close()
    """"""
    loudnes_db_smoothed = max_smooth(loudness_db, 128)
    loudnes_db_smoothed = avg_smooth(loudnes_db_smoothed, 128)
    """
    plt.figure(figsize=(10, 4))
    plt.plot(t, loudnes_db_smoothed, color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Loudness')
    plt.title('Loudness Curve (smoothed)')
    plt.show()
    plt.close()
    """
    max_db = np.max(loudnes_db_smoothed)
    threshold = max_db - 1.5 # -1.5dB
    is_drop = (loudnes_db_smoothed >= threshold).astype(int)
    """"""
    
    # num_frame_per_clip = librosa.time_to_frames(7.5, sr=sr)
    # 1 clip: 7.5 seconds (128bpm), 4 measures
    # 1 drop loop: 15 seconds (128bpm), 8 measures
    num_frame_per_droploop = librosa.time_to_frames(15, sr=sr)
    st, ed = -1, -1
    drop_sections = [] # unit is sample index
    
    while st < len(is_drop):
        st = st + 1
        while st < len(is_drop) and is_drop[st] == 0:
            st = st + 1
        if st == len(is_drop):
            break
        ed = st
        while ed < len(is_drop) and is_drop[ed] == 1:
            ed = ed + 1
        if ed - st >= num_frame_per_droploop:
            # print(f"Drop detected: {st} to {ed}")
            # print(st, ed)
            drop_sections += [(librosa.frames_to_time(st), librosa.frames_to_time(ed))]
            st = ed
        else:
            st = ed
    
    if debug:
        plt.figure(figsize=(10, 4))
        plt.plot(t, is_drop, color='b')
        plt.xlabel('Time (s)')
        plt.xticks(np.arange(0, max(t), 10))
        plt.title('Drop Indicator')
        for st, ed in drop_sections:
            plt.axvline(x=st, color='r', linestyle='--')
            plt.axvline(x=ed, color='g', linestyle='--')
        plt.show()
        plt.close()
    """"""
    
    return {
        'audio_path': audio_path,
        'drop_sections': drop_sections,
    }
    
    
    
if __name__ == "__main__":
    # ret = find_drop("/Users/ca7ax/Downloads/audio/Nothing To Hide - PeTE _ Trevor Omoto _ Peter Pentsak.ogg", True)
    # print(ret)
    # find_drop("/Users/ca7ax/Downloads/Martin Garrix - Now That I've Found You (feat. John & Michel) [Official Video].mp3")
    # find_drop("/Users/ca7ax/Downloads/Martin Garrix & Third Party - Lions In The Wild [Official Video].mp3")
    drop_annotations = []
    
    detected = 0
    
    for audio_name in tqdm(os.listdir(AUDIO_DIR)):
        audio_path = os.path.join(AUDIO_DIR, audio_name)
        annotation = find_drop(audio_path)
        drop_annotations += [annotation]
        detected += int(annotation['drop_sections'] != [])
        
    print(f"Detection rate: {detected / len(drop_annotations)}")
        
    with open('detected_drops.json', 'w') as f:
        json.dump(drop_annotations, f)