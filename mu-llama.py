"""
Please place this under your MU-LLaMA/MU-LLaMA path to do the inference on HouseX-v2.
Remember to modify the corresponding paths to MU-LLaMA components' checkpoints.
"""
import os, json
from tqdm import tqdm
import numpy as np
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import data.utils as data
import llama

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

ALL_GENRES = [
    'Progressive House',
    'Future House',
    'Bass House',
    'Tech House',
    'Deep House',
    'Bigroom',
    'Future Rave',
    'Slap House',
]

with open('/home/xinyu.li/housex-v2/misc/v2-doc.txt', 'r') as f:
    reference = ''.join(f.readlines())

llama_dir = "/home/xinyu.li/MuLLM-weights/MU-LLaMA/LLaMA"

model = llama.load("/home/xinyu.li/MuLLM-weights/MU-LLaMA/checkpoint.pth", llama_dir, knn=True, device="cuda")
model.eval()

if __name__ == '__main__':
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
        
        inputs = {}
        #image = data.load_and_transform_vision_data(["examples/girl.jpg"], device='cuda')
        #inputs['Image'] = [image, 1]
        audio = data.load_and_transform_audio_data([track_abs_path,],)
        inputs['Audio'] = [audio, 1]

        results = model.generate(
            inputs,
            [llama.format_prompt(
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
                Do not include any other information in your answer.'
            )],
            max_gen_len=256
        )
        result = results[0].strip()
        
        response = result
        
        pred_id = -1
        if len(re.findall(r'\d+', response)) > 0:  # the model gives a number
            pred_id = int(re.findall(r'\d+', response)[0]) - 1
        else: # the model gives a genre name
            for genre_id, genre in enumerate(ALL_GENRES):
                if genre.lower() in response:
                    pred_id = genre_id
                    break
        
        if pred_id == -1:
            continue
        
        label = clip_info['label']
        
        print(label, result)
        
        intersect_cnt += int(label[pred_id] > 0)
        accurate_cnt += int(pred_id == np.argmax(label))
        # print(label, response)
        all_preds += [pred_id]
        all_labels += [np.argmax(label)]        
    
    accuracy = accurate_cnt / len(clip_info_list)
    intersect_rate = intersect_cnt / len(clip_info_list)
    
    print("Accuracy:", accurate_cnt / len(clip_info_list))
    print("Intersect-rate:", intersect_cnt / len(clip_info_list))
    
    with open(os.path.join(clip_info_dir, 'mu-llama_result.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy}\nIntersect-rate: {intersect_rate}")
    
    print(compute_metrics(np.array(all_preds), np.array(all_labels)))


