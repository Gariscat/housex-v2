# 🔥 Mainstage Sub-Genre Benchmark 🔥
An extended classification benchmark with soft-labeling for mainstream dance music in the style of house, covering progressive house, future house, bass house, tech house, deep house, bigroom, future rave and slap house.

<!--The paper is on [arXiv](https://arxiv.org/abs/2409.06690).-->

## Definition

1. **Progressive House**: Characterized by its highly melodic structure and regular groove, this genre typically features a main sound composed of supersaw synths, piano, and strings. [An example](https://youtu.be/Lpjcm1F8tY8?feature=shared&t=90).
2. **Future House**: Defined by its fragmented rhythm and a less pronounced melody compared to Progressive House, this genre is distinguished by an overall tech-driven, futuristic sound. [An example](https://youtu.be/G4v_EPDxTcA?feature=shared&t=24).
3. **Bass House**: Characterized by the absence of a prominent lead instrument or the presence of a non-melodic lead, this genre features minimal chord progression, heavily distorted sounds, and a noisy texture, making it well-suited for outdoor music festivals. It often carries a dark, intense vibe. [An example](https://youtu.be/mTmet4jAkEA?si=CZofnG6wr5uq3iyA&t=24).
4. **Tech House**: Similar to Bass House but with a less noisy sound profile, this genre is more suitable for dance halls and bars, where the atmosphere is energetic but not as intense. [An example](https://youtu.be/nSG21KhzexU?si=mQqahlR_4UBQHCLy&t=21).
5. **Deep House**: Quieter than Tech House, this genre is characterized by a smooth groove, a strong atmospheric presence with large reverb, and a generally slower rhythm. [An example](https://youtu.be/2YA3yE3eO1w?feature=shared&t=53).
6. **Bigroom**: Often referred to as "Festival EDM," this genre exhibits simple, sometimes trivial musical structure, characterized by its raw, energetic, and hyped atmosphere. [An example](https://youtu.be/9vMh9f41pqE?feature=shared&t=46).
7. **Future Rave**: Marked by a relatively regular groove, this genre features a bass with moderate attack and a wave-like texture. The main sounds often exhibit quite electronic timbres. [An example](https://youtu.be/gvJQSAvA2yQ?feature=shared&t=67).
8. **Slap House**: A popular choice for BGM in short videos, this genre typically includes vocals and features a plucked bouncy bass. It often lacks chordal instruments, with the bass providing some mid and high frequencies. [An example](https://youtu.be/P0t8c9YwSM4?feature=shared&t=32).

Why soft labels? See an [example](https://www.youtube.com/watch?v=pISSIJCY_io) showing both progressive house and future house vibes. This multi-genre characteristic is in fact ubiquitous among EDM songs.

## Collection

Our dataset contains 1035 tracks. You can download it from [Google Drive](https://drive.google.com/drive/folders/12VNfriD5d6aUGN5w-LVKxT0Gsqe7RpEL?usp=sharing) (if the link expires, please contact xl3133@nyu.edu). We split them into 3 parts due to the file upload limit in ```Label-Studio``` and that the data is labeled by multiple experts. These 3 parts are **not** 3 folds of the dataset in the sense of train/val/test. The splits are created after loading all the tracks from the folders.

P.S. The ```annotations``` folder is deprecated. We keep annotations together with audio files in standlone directories.

## Annotation

Annotation is done using ```Label-Studio``` (MANY thanks to the developers!). We use soft labeling such that the probabilities of each sub-genre should sum up to 1. We also prepared a simple UI to check existing annotations (.csv). Run ```utils.py``` to view annotations of audio files in a folder. Before you run, please make sure you have modified ```anno_path``` and ```audio_dir``` in the .py file to your corresponding paths.

## Get Started

We recommend creating a conda environment. After activation, run ```pip install -r requirements.txt``` to install all the packages needed.

*Note that these dependencies only support our model. For MU-LLaMA, MusiLingo, the Qwen series and Kimi-Audio, please refer to their demo pages for installation. Put ```mu-llama.py```, ```musilingo.py``` and ```kimi-audio.py``` to their root directories to do inferences since these models are not native in HuggingFace so far.*

## Loudness-Based Drop Detection

Modify the directories in ```detect.py``` to the 3 folders of our data (or your own data with the same structure), Run ```python detect.py``` to detect the drops of tracks using rule-based algorithm by volume thresholds. The detected drops are stored as ```.json``` files in the folders respectively.

## Dataset Generation

Again, modify the directories in the ```main``` function of ```dataset.py``` to the corresponding folders. Then, run ```python dataset.py``` to generate the training set and the validation set (which is also the test set in our context). You can specify the ```--use_chroma``` and ```--mode``` parameter to determine whether to include chromagrams in the data representation and which type of label (soft/hard) to use.

## Training

Run ```python train.py```. Again, use the above mentioned 2 parameters to specify the dataset. Also, you can control other parameters like the network architecture and which GPU to use (Please refer to ```train.py``` for details). After training, you can run ```python vis_emb.py --force_run``` to visualize the embeddings with dimension reduction techniques like PCA, t-SNE and UMAP. Make sure that the checkpoint you load is trained from the dataset type that matches the arguments in ```vis_emb.py``` to avoid unexpected results. Checkpoints trained on dataset without chromagrams could produce random scatter points on dataset with chromagrams :|

## Deployment

The demo of stage visuals controlled by our classification model is [here](https://drive.google.com/drive/folders/1NJOy-fh-ozCiSy-olYjuxENNqti65o5T).

## Copyright
We are committed to ethical and legal research practices and have carefully considered copyright implications in our non-commercial, academic work aimed at advancing music information retrieval (MIR) for EDM. Below, we address these concerns and clarify our approach:

1. **Justification for Commercial Releases**: To ensure high audio quality and representativeness of contemporary EDM, we used commercial releases, as these reflect the production standards and diversity of the genre’s biggest hits. This choice strengthens the validity and generalizability of our findings, which aim to benefit the MIR community and, indirectly, the music industry through improved music analysis tools.

2. **Non-Commercial Academic Research**: Our study is purely academic, with no commercial intent or application. The dataset and model are developed solely to advance MIR techniques for EDM sub-genre classification, contributing to the broader scientific community’s understanding of music structure and style.

3. **Transformative Use of Limited Excerpts**: We extracted only the “drop” sections of the songs, which are short, distinct segments (typically 15–30 seconds). This use is transformative, as the drops are processed for feature extraction and classification, not for reproduction or consumption as music. The dataset does not enable reconstruction of the original songs, ensuring no substitution for the artists’ or labels’ original market.

4. **No Market Harm**: Our work, even if the dataset is made public, poses no threat to the commercial market of the original artists or record labels. The dataset consists of processed audio features and short excerpts, not full tracks, and is intended for research purposes only. To the best of our knowledge, they could not be used to replicate or compete with the original songs,

We are faithfully grateful to all the artists who produced these amazing tracks. Still, if you have copyright issues, please contact xl3133@nyu.edu.
