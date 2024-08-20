from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from dataset import HouseXDataset
from utils import read_audio_st_ed, compute_metrics
import os, json
from tqdm import tqdm
import numpy as np
import re
import random

torch.manual_seed(42)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()

"""print(response)
# The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".

# 2nd dialogue turn
response, history = model.chat(tokenizer, 'Find the start time and end time of the word the first drop, which is the chorus/climax for EDM music.', history=history)
print(response)"""

if __name__ == "__main__":
    with open('/root/autodl-tmp/HouseX-v2-doc.txt', 'r') as f:
        reference = ''.join(f.readlines())
    ### print(reference)
    
    clip_info_dir = '/root/autodl-tmp/standalone_test/'
    with open(os.path.join(clip_info_dir, 'clip_info.json'), 'r') as f:
        clip_info_list = json.load(f)
    
    ### random.shuffle(clip_info_list)
    ### clip_info_list = clip_info_list[:50]
    accurate_cnt = 0
    intersect_cnt = 0

    all_preds, all_labels = [], []
        
    for clip_info in tqdm(clip_info_list):
        track_name = os.path.basename(clip_info["track_path"])
        track_abs_path = os.path.join(clip_info_dir, track_name)
        query = tokenizer.from_list_format([
            {'audio': track_abs_path}, # Either a local path or an url
            {'text':
                f'From the perspective of an EDM producer, \
                we have some background knowledge for house music classification as references. \
                {reference} \
                What is the genre of this song? Answer to the best of your knowledge. \
                Please only output the number of the genre in the following list:\n\
                1. progressive house\n\
                2. future house/future bounce\n\
                3. bass house\n\
                4. tech house\n\
                5. bigroom\n\
                6. deep house\n\
                7. future rave\n\
                8. slap house/Brazilian bass\n. \
                Do not include any other information in your answer.'},
        ])
        response, history = model.chat(tokenizer, query=query, history=None)
        
        pred_id = -1
        if len(re.findall(r'\d+', response)) > 0:  # the model gives a number
            pred_id = int(re.findall(r'\d+', response)[0]) - 1
        else: # the model gives a genre name
            from config import ALL_GENRES
            for genre_id, genre in enumerate(ALL_GENRES):
                if genre.lower() in response:
                    pred_id = genre_id
                    break
        
        if pred_id == -1:
            continue
        
        label = clip_info['label']
        intersect_cnt += int(label[pred_id] > 0)
        accurate_cnt += int(pred_id == np.argmax(label))
        # print(label, response)
        all_preds += [pred_id]
        all_labels += [np.argmax(label)]        
    
    accuracy = accurate_cnt / len(clip_info_list)
    intersect_rate = intersect_cnt / len(clip_info_list)
    
    print("Accuracy:", accurate_cnt / len(clip_info_list))
    print("Intersect-rate:", intersect_cnt / len(clip_info_list))
    
    with open(os.path.join(clip_info_dir, 'qwen_result.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy}\nIntersect-rate: {intersect_rate}")
    
    print(compute_metrics(np.array(all_preds), np.array(all_labels)))
    """
        clip_info['qwen_resonse'] = response
        
    with open(os.path.join(clip_info_dir, 'qwen_answer.json'), 'w') as f:
        json.dump(clip_info_list, f)
    """
