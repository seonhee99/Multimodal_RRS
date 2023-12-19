import os, sys
import copy
import json
import pandas as pd
import numpy as np
from typing import List

import torch
from torch.cuda.amp import autocast
from llama import Llama
from transformers import LlamaTokenizer, LlamaModel, LlamaForCausalLM
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from rank_bm25 import BM25Okapi
from PIL import Image
import argparse
import random
import time
import fire
from tqdm import tqdm

SEED = 0
BATCH_SIZE = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_arguments():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    
    parser.add_argument("--data_type", type=str, required=True, help="Type of test data")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the image directory")
    parser.add_argument("--llama_path", type=str, required=True, help="Path to LLaMA weights")
    parser.add_argument("--prompt_path", type=str, required=True, help="Path to prompt from InstructBLIP")
    parser.add_argument("--save_file_path", type=str, required=True, help="Path to save the file")

    args = parser.parse_args()
    return args

def load_data(args, test_type, train_fraction=None, test_fraction=None, ):
    
    # load train data
    train_df = pd.read_csv(args.data_dir + f"data_train_full.csv")
    if train_fraction is not None:
        train_df = train_df.iloc[:train_fraction, :]
        
    image_paths = [p.split(',') for p in train_df.images]
    processed_image_paths = [[p.split('files/')[1] for p in paths] for paths in image_paths ]
    final_image_paths = [[os.path.join(args.image_dir, p) for p in paths] for paths in processed_image_paths]
        
    train_df['_processed_image_paths'] = processed_image_paths
    train_df['image_paths'] = final_image_paths

    # load test data
    test_df = pd.read_csv(args.data_dir + f"toy_data_test_{test_type}.csv")
    if test_fraction is not None:
        test_df = test_df.iloc[:test_fraction, :]
    
    image_paths = [p.split(',') for p in test_df.images]
    processed_image_paths = [[p.split('files/')[1] for p in paths] for paths in image_paths ]
    final_image_paths = [[os.path.join(args.image_dir, p) for p in paths] for paths in processed_image_paths]
    test_df['_processed_image_paths'] = processed_image_paths
    test_df['image_paths'] = final_image_paths
    
    return train_df, test_df

def load_oracle_annotation(df):
    annotation_path = "../data/summarization_annotations.json"
    with open(annotation_path) as f:
        annotations = json.load(f)
        
    all_annotations = annotations['train'] + annotations['val'] + annotations['test']
    annotation_dict = {x['image_path'][0] : x['report_label'] for x in all_annotations}
    label = [annotation_dict[xi] for xi in df.processed_image_paths]
    df['label'] = label
    
    return df
    

def init_retriever(findings):
    
    tokenized_corpus = [finding.split() for finding in findings]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def init_model(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.llama_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(args.llama_path)
    
    return tokenizer, model


def main(
    args,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    test_type: str = 'full',
):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device :", device)
    torch.cuda.empty_cache()
    
    tokenizer, model = init_model(args)
    model = model.to(device)
    print("LLaMA Loaded!")
    
    
    if args.data_type == 'all':
        test_types = ['full', 'corrupted_0.2', 'corrupted_0.5', 'corrupted_0.8']
    else:
        test_types = [args.data_type]
        
    for test_type in test_types:
        
        train_data, test_data = load_data(args, test_type=test_type)
        
        with open(args.prompt_path.format(test_type)) as f:
            all_prompt = json.load(f)
        
        all_results = dict()
        retriever = init_retriever(train_data.findings)

        for i,sample in enumerate(test_data.itertuples()): # itertuples for pd.dataframe
            
            st = time.time()
            
            _id = sample.image_paths[0]
            
            prompt = all_prompt[_id][test_type]
            prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            with autocast():
                results = model.generate(
                                prompt_tokens,
                                do_sample=True,
                                top_p=top_p,
                                temperature=1.0,
                                min_length=3,
                                max_length=len(prompt_tokens[0]) + 100,
                                top_k=50,
                                repetition_penalty=1.0,
                                length_penalty=1,
                            )
            output_text = tokenizer.decode(results[0], skip_special_tokens=True)
            print(prompt)
            print()
            print(output_text)
            processed_text = output_text[len(prompt):].split('\n\n')[0].strip()
            all_results[_id] = {
                'prompt' : prompt,
                'input_token_length' : prompt_tokens.shape[1],
                'gold' : sample.impressions,
                'pred' : processed_text,
                'raw_pred' : output_text,
                'time' : time.time() - st,
            }
            
            if i % BATCH_SIZE == 0:
                with open(args.save_file_path.format(test_type), 'w') as f:
                    json.dump(all_results, f)
                    
            del prompt_tokens, results
            torch.cuda.empty_cache()
                    
        with open(args.save_file_path.format(test_type), 'w') as f:
            json.dump(all_results, f)
            
        # del tokenizer, model
        torch.cuda.empty_cache()
        
        evaluate(all_results)
        

def evaluate(results):
    from rouge import Rouge 
    rouge = Rouge()
    
    # from radgraph import F1RadGraph
    # f1radgraph = F1RadGraph(reward_level="partial")
    
    generated_impressions = []
    reference_impressions = []
    
    null_string = []
    for k,v in results.items():
        
        if v['pred'] and len(set(v['pred']))>2:
            generated_impressions.append(v['pred'])
        else:
            null_string.append(k)
            generated_impressions.append('_')
            
        reference_impressions.append(v['gold'])
        
    print(" ====== RESULT ====== ")
    try:
        print(v['prompt'].split('\n')[0])
        print(f"Total {len(reference_impressions)} samples, with {len(null_string)} empty samples")

        score_rouge = rouge.get_scores(generated_impressions, reference_impressions, avg=True)
        print("ROUGE-1:", score_rouge["rouge-1"])
        print("ROUGE-2:", score_rouge["rouge-2"])
        print("ROUGE-L:", score_rouge["rouge-l"])
    
        score_f1radgraph, _, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=generated_impressions, refs=reference_impressions)

        print("F1RadGraph:", score_f1radgraph)
        print()
    
        del generated_impressions, reference_impressions, score_rouge, 
        del score_f1radgraph, _, hypothesis_annotation_lists, reference_annotation_lists
        torch.cuda.empty_cache()
    except:
        print("Evaluation error")
    

if __name__ == "__main__":
    args = get_arguments()
    main(args)