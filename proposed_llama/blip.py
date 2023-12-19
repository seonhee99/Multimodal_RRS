import os, sys
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from rank_bm25 import BM25Okapi
from PIL import Image
import copy
import requests
import random
import torch

def get_arguments():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    
    parser.add_argument("--data_type", type=str, required=True, help="Type of test data")
    parser.add_argument("--use_oracle", type=bool, required=True, help="To use oracle image description")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the image directory")
    parser.add_argument("--instruct_blip_path", type=str, required=True, help="Path to InstructBlip weights")
    parser.add_argument("--description_path", type=str, required=True, help="Path to disease descriptions")
    parser.add_argument("--save_file_path", type=str, required=True, help="Path to save the file")
    parser.add_argument("--save_prompt_path", type=str, required=True, help="Path to save the prompt")
    parser.add_argument("--agg_type", type=str, choices=['avg', 'sum', 'none'], required=True, help="Image aggregation method")

    args = parser.parse_args()
    return args

# VQAs_path = "prompt/disease_descriptor_prompt.json"
# image_dir = "/nfs_data_storage/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
# data_dir = "/nfs_edlab/shcho/lec/idl/project/data/"
# save_file_path = "generation/descriptor_explain3.json"
# save_prompt_path = "prompt/descriptor_explain3.json"
batch_size = 16
MAX_SEQ_LEN = 2000
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    
    args = get_arguments()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device :", device)
    
    train_data, test_data = load_data(args, train_fraction=None, test_fraction=None)
    
    model, processor, descriptions = set_vlm_and_prompt(args, device)
    
    # 그냥 test set에 대해서만 돌릴 때.
    # results = run_blip(args, model, processor, test_data, descriptions)

    results = run_blip_for_llama(args, model, processor, train_data, test_data, descriptions)
    

def set_vlm_and_prompt(args, device):
    
    if args.use_oracle:
        print("skip loading BLIP")
        model = None
        processor = None
        
    else:
        st = time.time()
        model = InstructBlipForConditionalGeneration.from_pretrained(args.instruct_blip_path)
        model = model.to(device)
        processor = InstructBlipProcessor.from_pretrained(args.instruct_blip_path)
        print(time.time() - st, "secs to load InstructBlip model and processor")
        
    with open(args.description_path) as f:
        descriptor_prompt = json.load(f)
    return model, processor, descriptor_prompt



def run_blip_for_llama(args, model, processor, train_data, test_data, descriptions, verbose=False ):
    '''
    model : InstructBlipForConditionalGeneration
    processor : InstructBlipProcessor
    '''
    
    results = {'prompt' : dict(), 'blip' : dict() }
    
    if args.data_type == 'all':
        test_types = ['full', 'corrupted_0.2', 'corrupted_0.5', 'corrupted_0.8']
    else:
        test_types = [args.data_type]
    
    retriever = { test_type: init_retriever(train_data[test_type].findings) for test_type in test_types}
    
    for test_type in test_types:
        
        for i,sample in enumerate(test_data[test_type].itertuples()):

            _id = sample.image_paths[0]
            print("Sample", _id, end=' ')

            st = time.time()

            results['blip'][_id] = {test_type : dict() for test_type in test_types}

            llama_prompt = 'Please provide a concise summary of the following chest X-ray report. The text description of the X-ray images and the full report, named "Finding" will be provided. Your task is to focus on the key findings and diagnoses noted primarily in the image description, while also incorporating relevant details from the full report. The summary, named "Impression", should be concise and presented in correct English sentences.\n'

            llama_prompt_dict = {test_type: llama_prompt for test_type in test_types}
            blip_output_dict = {test_type: dict() for test_type in test_types}

            # Dynamic few-shot
            for test_type in test_types:
                sample = test_data[test_type].iloc[i]
                retrieved_idxes = _run_retriever(sample, retriever[test_type], k=2)

                for ri, idx in enumerate(retrieved_idxes):

                    if not args.use_oracle:
                        blip_output = _blip_generate(model, processor, descriptions, train_data[test_type].image_paths[idx], args.agg_type)
                        blip_output_dict[test_type][f'retrieved_sample{ri}'] = {
                            'findings': train_data['full'].findings[idx],
                            'impressions': train_data['full'].impressions[idx],
                            'blip_inference':blip_output,
                            'time' : time.time() - st
                        }

                    _prompt = "Image description: "
                    if args.use_oracle:
                        annotation = train_data[test_type].label.iloc[idx]
                        _prompt += _get_oracle_description(train_data[test_type].image_paths[idx][0], annotation, descriptions)
                    else:
                        _prompt += _get_image_description_for_llama_prompt(blip_output)
                    _prompt += f"Finding: {train_data[test_type].findings.iloc[idx]}\n"
                    _prompt += f"Impression: {train_data[test_type].impressions.iloc[idx]}\n"
                    _prompt += f"\n"
                    llama_prompt_dict[test_type] += _prompt

            # test sample
            if not args.use_oracle:
                blip_output = _blip_generate(model, processor, descriptions, sample.image_paths, args.agg_type)
                blip_output_dict['test_sample'] = {
                    'findings': sample.findings,
                    'impressions': sample.impressions,
                    'blip_inference':blip_output,
                    'time' : time.time() - st
                }

            for test_type in test_types:
                sample = test_data[test_type].iloc[i]
                _prompt = "Image description: "
                if args.use_oracle:
                    annotation = sample.label
                    _prompt += _get_oracle_description(sample.image_paths[0], annotation, descriptions)
                else:
                    _prompt += _get_image_description_for_llama_prompt(blip_output)
                _prompt += f"Finding: {sample.findings}\n"
                _prompt += f"Impression: "
                llama_prompt_dict[test_type] += _prompt

            results['prompt'][_id] = llama_prompt_dict
            if not args.use_oracle:
                results['blip'][_id] = blip_output_dict
                with open(args.save_file_path.format(test_type), 'w') as f:
                    json.dump(results['blip'], f)

            with open(args.save_prompt_path.format(test_type), 'w') as f:
                json.dump(results['prompt'], f)
            print("done")
            print(time.time() - st, "secs")
        
        
    return results
    

def _blip_generate(model, processor, description_dict, image_paths, agg_type):
    
    device = model.device
    if agg_type == 'none':
        image_path = image_paths[0]
        
    loaded_image = Image.open(image_path).convert("RGB")
    blip_prompt = 'Are there any {} in the image? Explain in one sentence.'
    blip_output = dict()
    
    for disease, descriptor_list in description_dict.items():
        
        blip_output_disease = dict()
        is_descriptor = 0

        for descriptor in descriptor_list:
            
            text_prompt = blip_prompt.format(descriptor)
            
            inputs = processor(images=loaded_image, text=text_prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, do_sample=True, num_beams=5, max_length=MAX_SEQ_LEN, min_length=1, top_p=0.9, repetition_penalty=1.5, temperature=1 )
            decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            blip_output_disease[descriptor] = decoded_outputs

            del inputs, outputs
            torch.cuda.empty_cache()
        
        blip_output[disease] = blip_output_disease
    
    return blip_output
    
    
def _get_image_description_for_llama_prompt(blip_output):
    '''
    result : dict
    { disease(str) : { descriptor(str) : blip_output(str), ... }, .. }
    
    Given the result dictionary, return aggregated llama prompt.
    '''
    prompt = ""
    for disease, _blip_output in blip_output.items():
        prompt += f"{disease} - "
        descriptor_cnt = 0
        for di, (descriptor, generated_text) in enumerate(_blip_output.items()):
            ### YES, NO만 넣는 세팅 ###
            answer = generated_text.split()[0].strip(',').lower()
            prompt += f"{descriptor}" if answer == 'yes' else f"{answer} {descriptor}"
            prompt += ", " if (di != len(_blip_output)-1) else ". "

            #### 그래도 반환된 문장을 다 넣는 세팅 ###
            # prompt += generated_text
            # prompt += " "
            
            descriptor_cnt += 1 if generated_text.lower().startswith('yes') else 0
    
        if descriptor_cnt > len(_blip_output) * 0.5:
            prompt += f"In general, it seems there is {disease} in the image.\n"
        else:
            prompt += f"In general, it seems there is no {disease} in the image.\n"
    
    return prompt

def _get_oracle_description(image_path, annotation, descriptor_prompt):
    loaded_image = Image.open(image_path).convert("RGB")
    prompt = 'Are there any {} in the image? Explain in one sentence.'
    description = ''
    
    for disease, descriptors in descriptor_prompt.items():
        description += f"{disease} - "
        descriptor_boolean_yes = 0
        
        if disease.capitalize() in annotation:
            bool_tag = 'yes'
        else:
            bool_tag = 'no'
        for di, desc in enumerate(descriptors):
            if bool_tag == 'yes':
                description += f"{desc}"
            else:
                description += f"{bool_tag} {desc}"
                
            if di != len(descriptors) - 1:
                description += ", "
            else:
                description += "\n"

        if bool_tag == 'yes':
            description += f"In general, it seems there is {disease} in the image.\n"
        else:
            description += f"In general, it seems there is no {disease} in the image.\n"
            
    return description


def init_retriever(findings):
    tokenized_corpus = [finding.split() for finding in findings]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def _run_retriever(sample, retriever, k=2):
    scores = retriever.get_scores(sample.findings.split())
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    top_indices = [idx for idx in sorted_indices][:k]
    return top_indices


def load_oracle_annotation(df):
    annotation_path = "/nfs_edlab/shcho/multimodal_rrs/data/summarization_annotations.json"
    with open(annotation_path) as f:
        annotations = json.load(f)
    # import pdb; pdb.set_trace()
    all_annotations = annotations['train'] + annotations['val'] + annotations['test']
    annotation_dict = {x['image_path'][0] : x['report_label'] for x in all_annotations}
    
    label = [annotation_dict[xi[0]] for xi in df.processed_image_paths]
    df['label'] = label
    
    return df


def load_data(args, train_fraction=None, test_fraction=None):  
    
    train_data = dict()
    test_data = dict()
    
    for test_type in ['full', 'corrupted_0.2', 'corrupted_0.5', 'corrupted_0.8']: 
        
        # load train data
        train_df = pd.read_csv(args.data_dir + f"data_train_full.csv")
        if train_fraction is not None:
            train_df = train_df.iloc[:train_fraction, :]
        # image_paths = [p.split(',')[0] for p in train_df.images]
        # processed_image_paths = [p.split('files/')[1] for p in image_paths]
        # final_image_paths = [os.path.join(args.image_dir, p) for p in processed_image_paths]
        
        image_paths = [p.split(',') for p in train_df.images]
        processed_image_paths = [[p.split('files/')[1] for p in paths] for paths in image_paths ]
        final_image_paths = [[os.path.join(args.image_dir, p) for p in paths] for paths in processed_image_paths]
        
        train_df['processed_image_paths'] = processed_image_paths
        train_df['image_paths'] = final_image_paths
        
        if args.use_oracle:
            train_df = load_oracle_annotation(train_df)
        
        train_data[test_type] = train_df
        
        # load test data
        test_df = pd.read_csv(args.data_dir + f"toy_data_test_{test_type}.csv")
        if test_fraction is not None:
            test_df = test_df.iloc[:test_fraction, :]
            
        # image_paths = [p.split(',')[0] for p in test_df.images]
        # processed_image_paths = [p.split('files/')[1] for p in image_paths]
        # final_image_paths = [os.path.join(args.image_dir, p) for p in processed_image_paths]
        
        image_paths = [p.split(',') for p in test_df.images]
        processed_image_paths = [[p.split('files/')[1] for p in paths] for paths in image_paths ]
        final_image_paths = [[os.path.join(args.image_dir, p) for p in paths] for paths in processed_image_paths]
        
        test_df['processed_image_paths'] = processed_image_paths
        test_df['image_paths'] = final_image_paths
        if args.use_oracle:
            test_df = load_oracle_annotation(test_df)
        
        test_data[test_type] = test_df
    
    return train_data, test_data

if __name__ == '__main__':
    main()