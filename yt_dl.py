import os
from pytube import YouTube
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

# Directory to save the downloaded and processed files
download_dir = "downloaded_videos"
audio_dir = "processed_audio"

# Create directories if they don't exist
os.makedirs(download_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)

# Function to download a YouTube video and return the path to the downloaded file
def download_video(url, download_dir):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=False).first()
    out_file = video.download(output_path=download_dir)
    return out_file

# Function to extract, resample, and save the audio
def process_audio(video_path, audio_dir, sample_rate=44100):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = os.path.join(audio_dir, os.path.basename(video_path).replace(".mp4", ".ogg"))
    
    # Save the audio to a temporary wav file
    temp_wav_path = audio_path.replace(".ogg", ".wav")
    audio.write_audiofile(temp_wav_path, codec='pcm_s16le')

    # Load the audio using pydub
    audio_segment = AudioSegment.from_wav(temp_wav_path)
    
    # Resample the audio
    audio_segment = audio_segment.set_frame_rate(sample_rate)
    
    # Save as Ogg
    audio_segment.export(audio_path, format="ogg")

    # Remove the temporary wav file
    os.remove(temp_wav_path)

    # Close video file
    video.close()

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
    input_file = "input.txt"  # Replace with your input file path
    process_file(input_file)
