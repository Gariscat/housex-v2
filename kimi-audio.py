import torch
import os, json
from tqdm import tqdm
import numpy as np
import re
import random
import librosa
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(42)

from kimia_infer.api.kimia import KimiAudio
import os
import soundfile as sf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_metrics(predictions, targets, average='weighted'):
    """
    Computes accuracy, precision, recall, F1-score, and confusion matrix for a classification problem.

    Parameters:
    - predictions (np.array): Predicted labels.
    - targets (np.array): True labels.
    - average (str): Averaging strategy for precision, recall, and F1-score ('binary', 'micro', 'macro', 'weighted').

    Returns:
    - dict: A dictionary containing the computed metrics.
    """

    # Compute accuracy
    accuracy = accuracy_score(targets, predictions)

    # Compute precision, recall, and F1-score
    precision = precision_score(targets, predictions, average=average)
    recall = recall_score(targets, predictions, average=average)
    f1 = f1_score(targets, predictions, average=average)

    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions)

    # Return the metrics as a dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }

if __name__ == "__main__":

    model = KimiAudio(
        model_path="moonshotai/Kimi-Audio-7B-Instruct",
        load_detokenizer=True,
    )

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }

    with open('../housex-v2/misc/v2-doc.txt', 'r') as f:
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
        messages = [
            {"role": "user", "message_type": "text", "content": f'What is the genre of this song? Answer to the best of your knowledge. \
                Please first describe the music. Then, output the number of the genre in the following list:\n\
                1. progressive house\n\
                2. future house/future bounce\n\
                3. bass house\n\
                4. tech house\n\
                5. bigroom\n\
                6. deep house\n\
                7. future rave\n\
                8. slap house/Brazilian bass\n\
                For your information, we provide some reference.\n\
                {reference}'},
            {
                "role": "user",
                "message_type": "audio",
                "content": track_abs_path,
            },
        ]
        # conversation = [
        #     {'role': 'system', 'content': [
        #         {"type": "text", "text": 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."'},
        #     ]}, 
        #     {"role": "user", "content": [
        #         {"type": "audio", "audio": track_abs_path},
        #         {"type": "text", "text": None,
        #     ]},
        # ]
        # text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        # inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        # inputs = inputs.to(model.device).to(model.dtype)

        # # Inference: Generation of the output text and audio
        # text_ids, audio = model.generate(**inputs, max_length=2048, use_audio_in_video=False)
        
        # response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print("-------------")
        # print(type(response))
        # print(response)
        # print("-------------")
        wav, response = model.generate(messages, **sampling_params, output_type="text")
        with open(f"test_audios/{track_name.replace('.ogg', '.txt')}", 'w') as f:
            f.write(response)
        # response = response.split("assistant")[-1]
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
    
    with open(os.path.join(clip_info_dir, 'kimi_audio.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy}\nIntersect-rate: {intersect_rate}")
    
    print(compute_metrics(np.array(all_preds), np.array(all_labels)))
    """
        clip_info['qwen_resonse'] = response
        
    with open(os.path.join(clip_info_dir, 'qwen_answer.json'), 'w') as f:
        json.dump(clip_info_list, f)
    """
