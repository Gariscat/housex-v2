import unicodedata
import os
import unicodedata
import os
from pypinyin import pinyin, Style
from korean_romanizer.romanizer import Romanizer
import pykakasi

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
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            rename_file_to_normalized(os.path.join(directory, filename), debug=debug)
            
            
if __name__ == "__main__":
    normalize_files_in_directory("/home/ecs-user/Downloads/audio/", debug=True)
