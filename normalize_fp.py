import unicodedata
import os
import unicodedata
import os
from pypinyin import pinyin, Style
from korean_romanizer.romanizer import Romanizer
import pykakasi
import soundfile as sf
from tqdm import tqdm
import numpy as np
import os
from pydub import AudioSegment


def normalize_filename(filename: str):
    """
    # convert Japanese characters to Romanized
    filename = pykakasi.kakasi().convert(filename)[0]['passport']
    """
    # Convert Korean characters to Romanized
    filename = Romanizer(filename).romanize()
    # Convert Chinese characters to Pinyin
    filename = ''.join([_[0] for _ in pinyin(filename, style=Style.NORMAL)])
    
    normalized_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('utf-8')
    return normalized_filename

def rename_file_to_normalized(filename: str, debug: bool=False):
    normalized_filename = normalize_filename(filename)
    directory = os.path.dirname(filename)
    new_filename = os.path.join(directory, normalized_filename)
    
    if debug:
        print(f"Renaming:\n{filename}\nto\n{new_filename}")
    
    os.rename(filename, new_filename)
    
def normalize_files_in_directory(directory: str, debug: bool=False):
    
    """def convert_mp3_to_ogg(filename: str):
        # Load the audio file
        # audio, sr = sf.read(filename, always_2d=True)
        sound = AudioSegment.from_file(filename, format="mp3")
        # Create the output filename by replacing the file extension
        output_filename = filename.replace('.mp3', '.ogg')
        
        # Save the audio as an ogg file
        sound.export(output_filename, format="ogg")
        del sound"""
    
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            rename_file_to_normalized(os.path.join(directory, filename), debug=debug)
    """
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.mp3'):
            print(f"Converting {filename}")
            convert_mp3_to_ogg(os.path.join(directory, filename))
    """
                
if __name__ == "__main__":
    directory = "/Users/ca7ax/housex-v2/housex-v2.1-raw-data/"
    normalize_files_in_directory(directory, debug=True)
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.ogg'):
            count += 1

    print(f"Number of ogg files in the directory: {count}")