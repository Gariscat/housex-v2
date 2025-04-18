"""
Please follow the official huggingface page of Qwen2.5-Omni to prepare the environment (transformers from source, etc.).
"""
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from transformers.generation import GenerationConfig
import torch
from utils import read_audio_st_ed, compute_metrics
import os, json
from tqdm import tqdm
import numpy as np
import re
import random
import librosa
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(42)

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# default: Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-Omni-7B",
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

"""print(response)
# The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".

# 2nd dialogue turn
response, history = model.chat(tokenizer, 'Find the start time and end time of the word the first drop, which is the chorus/climax for EDM music.', history=history)
print(response)"""

if __name__ == "__main__":
    with open('./misc/v2-doc.txt', 'r') as f:
        reference = ''.join(f.readlines())
    ### print(reference)
    
    clip_info_dir = '/home/xinyu.li/autodl-tmp/standalone_test/'
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
        conversation = [
            {'role': 'system', 'content': [
                {"type": "text", "text": 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."'},
            ]}, 
            {"role": "user", "content": [
                {"type": "audio", "audio": track_abs_path},
                {"type": "text", "text": f'From the perspective of an EDM producer, \
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
            ]},
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)

        # Inference: Generation of the output text and audio
        text_ids, audio = model.generate(**inputs, max_length=2048, use_audio_in_video=False)
        
        response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print("-------------")
        # print(type(response))
        # print(response)
        # print("-------------")
        response = response.split("assistant")[-1]
        print(response)
        
        pred_id = -1
        if len(re.findall(r'\d+', response)) > 0:  # the model gives a number
            pred_id = int(re.findall(r'\d+', response)[0]) - 1
        else: # the model gives a genre name
            from config import ALL_GENRES
            for genre_id, genre in enumerate(ALL_GENRES):
                if genre.lower() in response:
                    pred_id = genre_id
                    break
        
        label = clip_info['label']
        
        if pred_id == -1:
            all_preds += [np.random.randint(0, len(ALL_GENRES))]
            all_labels += [np.argmax(label)]
            continue
        
        intersect_cnt += int(label[pred_id] > 0)
        accurate_cnt += int(pred_id == np.argmax(label))
        # print(label, response)
        all_preds += [pred_id]
        all_labels += [np.argmax(label)]        
    
    accuracy = accurate_cnt / len(clip_info_list)
    intersect_rate = intersect_cnt / len(clip_info_list)
    
    print("Accuracy:", accurate_cnt / len(clip_info_list))
    print("Intersect-rate:", intersect_cnt / len(clip_info_list))
    
    with open(os.path.join(clip_info_dir, 'qwen_audio_result.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy}\nIntersect-rate: {intersect_rate}")
    
    print(compute_metrics(np.array(all_preds), np.array(all_labels)))
    """
        clip_info['qwen_resonse'] = response
        
    with open(os.path.join(clip_info_dir, 'qwen_answer.json'), 'w') as f:
        json.dump(clip_info_list, f)
    """
