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

torch.manual_seed(42)

# Note: The default behavior now has injection attack prevention off.
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="cuda").eval()

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
            {'role': 'system', 'content': 'You are a helpful assistant.'}, 
            {"role": "user", "content": [
                {"type": "audio", "audio_url": track_abs_path},
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
        audios = [librosa.load(track_abs_path, sr=processor.feature_extractor.sampling_rate)[0],]
        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs["input_ids"] = inputs["input_ids"].to("cuda")
        inputs.input_ids = inputs.input_ids.to("cuda")
        
        generate_ids = model.generate(**inputs, max_length=2048)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
    
    with open(os.path.join(clip_info_dir, 'qwen2_audio_result.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy}\nIntersect-rate: {intersect_rate}")
    
    print(compute_metrics(np.array(all_preds), np.array(all_labels)))
    """
        clip_info['qwen_resonse'] = response
        
    with open(os.path.join(clip_info_dir, 'qwen_answer.json'), 'w') as f:
        json.dump(clip_info_list, f)
    """
