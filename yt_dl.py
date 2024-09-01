import os
import subprocess
from pydub import AudioSegment
from yt_dlp import YoutubeDL

# Directory to save the downloaded and processed files
download_dir = "/Users/admin/.downloaded_videos"
audio_dir = "/Users/admin/.processed_audio"

# Create directories if they don't exist
os.makedirs(download_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)

# Function to download a YouTube video using youtube-dl
def download_video(url, download_dir):
    with YoutubeDL() as ydl:
        ydl.download([url])

# Function to extract, resample, and save the audio
def process_audio(video_path, audio_dir, sample_rate=44100):
    # Load the video using pydub
    audio_segment = AudioSegment.from_file(video_path)
    
    # Resample the audio
    audio_segment = audio_segment.set_frame_rate(sample_rate)
    
    # Set the output path for the Ogg file
    audio_path = os.path.join(audio_dir, os.path.basename(video_path).replace(".mp4", ".ogg").replace(".webm", ".ogg"))
    
    # Save as Ogg
    audio_segment.export(audio_path, format="ogg")

# Main processing function
def process_file(input_file):
    with open(input_file, "r") as file:
        for line in file:
            url = line.strip()
            if url.startswith("http"):
                try:
                    video_path = download_video(url, download_dir)
                    process_audio(video_path, audio_dir)
                except Exception as e:
                    print(f"Failed to process {url}: {e}")

if __name__ == "__main__":
    input_file = "misc/s1_supplement_list.txt"  # Replace with your input file path
    process_file(input_file)
