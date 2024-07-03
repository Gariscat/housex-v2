import tkinter as tk
from tkinter import filedialog
import pygame
import threading
import pandas as pd
import os
from config import ALL_GENRES

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
        

if __name__ == '__main__':
    """ap = "/Users/ca7ax/Downloads/audio-100-test/Where You Wanna Be - R3HAB _ Elena Temnikova.ogg"
    AudioPlayer(ap)"""
    
    check_annotations(
        anno_path="/Users/ca7ax/Downloads/project-4-at-2024-07-03-13-33-8a6c2a33.csv",
        audio_dir="/Users/ca7ax/Library/Application Support/label-studio/media/upload/4"
    )
    