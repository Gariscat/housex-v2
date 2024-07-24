import tkinter as tk
import os
from config import ALL_GENRES, FRAME_LENGTH
import matplotlib.pyplot as plt
import librosa
import numpy as np
from argparse import ArgumentParser
import json
import soundfile as sf

class AudioPlayer:
    """
    A class that represents an audio player.

    Attributes:
        filepath (str): The path to the audio file.
        comment (str): Additional comments about the audio file.

    Methods:
        __init__(self, filepath: str, comment: str=''): Initializes the AudioPlayer object.
        setup_ui(self, comment: str=''): Sets up the user interface for the audio player.
        start_drag(self, event): Handles the start of dragging the progress bar.
        stop_drag(self, event): Handles the stop of dragging the progress bar.
        play_audio(self): Plays the audio file.
        pause_audio(self): Pauses the audio playback.
        stop_audio(self): Stops the audio playback.
        set_position(self, value): Sets the position of the audio playback.
        update_progress(self): Updates the progress of the audio playback.

    """
    
    def __init__(self, filepath: str, comment: str=''):
        """
        Initializes the AudioPlayer object.

        Args:
            filepath (str): The path to the audio file.
            comment (str, optional): Additional comments about the audio file. Defaults to ''.

        """
        import pygame
        
        self.filepath = filepath
        pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        self.playing = False
        self.paused = False
        self.user_dragging = False

        self.root = tk.Tk()
        self.setup_ui(comment)
        self.root.mainloop()

    def setup_ui(self, comment: str=''):
        """
        Sets up the user interface for the audio player.

        Args:
            comment (str, optional): Additional comments about the audio file. Defaults to ''.

        """
        self.root.title("Audio Player")

        # Title Label
        self.title_label = tk.Label(self.root, text=self.filepath.split("/")[-1])
        self.title_label.pack(pady=10)

        # Comment Section
        self.comment_label = tk.Label(self.root, text=f"Comments:\n{comment}")
        self.comment_label.pack(pady=5)
        self.comment_text = tk.Text(self.root, height=5, width=50)
        self.comment_text.pack(pady=5)

        # Progress Bar
        self.progress = tk.Scale(self.root, from_=0, to=pygame.mixer.Sound(self.filepath).get_length(), orient=tk.HORIZONTAL, length=400, command=self.set_position)
        self.progress.pack(pady=20)

        # Control Buttons
        self.play_button = tk.Button(self.root, text="Play", command=self.play_audio)
        self.play_button.pack(side=tk.LEFT, padx=10)
        self.pause_button = tk.Button(self.root, text="Pause", command=self.pause_audio)
        self.pause_button.pack(side=tk.LEFT, padx=10)
        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_audio)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.progress.bind("<ButtonPress-1>", self.start_drag)
        self.progress.bind("<ButtonRelease-1>", self.stop_drag)

        self.update_progress()

    def start_drag(self, event):
        """
        Handles the start of dragging the progress bar.

        Args:
            event: The event object.

        """
        self.user_dragging = True

    def stop_drag(self, event):
        """
        Handles the stop of dragging the progress bar.

        Args:
            event: The event object.

        """
        self.user_dragging = False

    def play_audio(self):
        """
        Plays the audio file.

        """
        if not self.playing:
            pygame.mixer.music.play()
            self.playing = True
        elif self.paused:
            pygame.mixer.music.unpause()
        self.paused = False

    def pause_audio(self):
        """
        Pauses the audio playback.

        """
        pygame.mixer.music.pause()
        self.paused = True

    def stop_audio(self):
        """
        Stops the audio playback.

        """
        pygame.mixer.music.stop()
        self.playing = False
        self.paused = False
        self.progress.set(0)

    def set_position(self, value):
        """
        Sets the position of the audio playback.

        Args:
            value: The position value.

        """
        if self.user_dragging:
            pygame.mixer.music.play(start=float(value))
            self.paused = False

    def update_progress(self):
        """
        Updates the progress of the audio playback.

        """
        if self.playing and not self.paused and not self.user_dragging:
            current_pos = pygame.mixer.music.get_pos() / 1000.0
            self.progress.set(current_pos)
        self.root.after(1000, self.update_progress)
        
        
def check_annotations(anno_path: str, audio_dir: str):
    """
    Check annotations for audio files.

    Args:
        anno_path (str): The path to the annotation file (CSV format).
        audio_dir (str): The directory containing the audio files.

    Returns:
        None
    """
    import pandas as pd
    assert anno_path.endswith(".csv")
    df = pd.read_csv(anno_path)
    for i in range(len(df)):
        row = df.iloc[i]
        audio_path = os.path.join(audio_dir, os.path.basename(row['audio']))
        
        comment = ''
        for genre in ALL_GENRES:
            # print(row[genre])
            score = float(row[genre][11:-2])
            if score > 0:
                comment = comment + f"{genre}: {score}\n"
        
        AudioPlayer(audio_path, comment)
        

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


def read_audio(audio_path: str, chunk_size: int=8192):
    y = []
    
    with sf.SoundFile(audio_path, 'r') as f:
        samplerate = f.samplerate
        channels = f.channels
        """print(f'Sample rate: {samplerate}')
        print(f'Channels: {channels}')"""

        while True:
            # Read a chunk of data
            chunk = f.read(chunk_size)
            if len(chunk) == 0:
                break
            if channels == 2:
                chunk = chunk.mean(axis=-1)
            y += chunk.tolist()
            
    return np.array(y), samplerate


def read_audio_st_ed(audio_path: str, st: float, ed: float):
    sr = librosa.get_samplerate(audio_path)
    y, _ = sf.read(audio_path, start=int(st*sr), stop=int(ed*sr))
    if len(y.shape) == 2:
        y = y.mean(axis=1)
    return y, sr


def find_drop(audio_path: str, debug: bool=False, left_trunc_sec: float=15, write_to_tmp: bool=True):
    """Detects the drop in an EDM track.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        sections (List[str]): pairs of (st, ed) of drops
    """
    y, sr = librosa.load(audio_path)
    # y, sr = read_audio(audio_path)
    y = y[int(left_trunc_sec*sr):]  # truncate the first x seconds (speed up later process)
    loudness = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH).flatten()
    # Convert loudness from rms to dB
    loudness = librosa.amplitude_to_db(loudness, ref=np.max)
    if debug:
        frames = range(len(loudness))
        t = librosa.frames_to_time(frames, sr=sr)
        # Plot the loudness curve
        plt.figure(figsize=(10, 4))
        plt.plot(t, loudness, color='b')
        plt.xlabel('Time (s)')
        plt.ylabel('Loudness')
        plt.title('Loudness Curve')
        plt.savefig(f'/root/{os.path.basename(audio_path)}_chunk.jpg')
        plt.close()
    """"""
    loudness = max_smooth(loudness, 128)
    loudness = avg_smooth(loudness, 128)
    """
    plt.figure(figsize=(10, 4))
    plt.plot(t, loudnes_db_smoothed, color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Loudness')
    plt.title('Loudness Curve (smoothed)')
    plt.show()
    plt.close()
    """
    max_db = np.max(loudness)
    threshold = max_db - 1.5 # -1.5dB
    is_drop = (loudness >= threshold).astype(int)
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
            drop_sections += [(
                librosa.frames_to_time(st)+left_trunc_sec,
                librosa.frames_to_time(ed)+left_trunc_sec
            )]
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
    ret = {
        'audio_path': audio_path,
        'drop_sections': drop_sections,
    }
    if not write_to_tmp:
        return ret
    with open('tmp.json', 'w') as f:
        json.dump(ret, f)


if __name__ == '__main__':
    # This part is a showcase for the UI for reading annotations
    """ap = "/Users/ca7ax/Downloads/audio-100-test/Where You Wanna Be - R3HAB _ Elena Temnikova.ogg"
    AudioPlayer(ap)"""
    
    """check_annotations(
        anno_path="/Users/ca7ax/Downloads/project-4-at-2024-07-03-13-33-8a6c2a33.csv",
        audio_dir="/Users/ca7ax/Library/Application Support/label-studio/media/upload/4"
    )"""
    # This part acts as a process to find the drop of a given track.
    # The reason why we do not call find_drop multiple times in detect.py
    # is to avoid RAM issues, possibly related to soundfile/librosa caching.
    parser = ArgumentParser()
    parser.add_argument('--audio_path', type=str, required=True)
    args = parser.parse_args()
    
    find_drop(args.audio_path)
    