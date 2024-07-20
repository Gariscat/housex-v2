import os
import shutil
import random

def shard_files(directory, n):
    # Create n separate folders
    for i in range(n):
        os.makedirs(f"{directory}/partition_{i+1}", exist_ok=True)

    # Get all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Shuffle the files
    random.shuffle(files)

    # Calculate the number of files per partition
    files_per_partition = len(files) // n + 1

    # Shard the files into partitions
    for i, file in enumerate(files):
        partition = i // files_per_partition
        shutil.move(f"{directory}/{file}", f"{directory}/partition_{partition+1}/{file}")

# Example usage
directory = "./housex-v2.1-raw-data"
n = 20  # Number of partitions
shard_files(directory, n)