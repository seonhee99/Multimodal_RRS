import torch
from PIL import Image
from transformers import AutoTokenizer
import random
import time
import pandas as pd
import json
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms
from rank_bm25 import BM25Okapi

# Set a random seed for reproducibility
random.seed(42)

# Function to load data from a CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    findings = df['findings'].tolist()
    impressions = df['impressions'].tolist()  
    image_paths = ["/home/choonghan/practice/" + path.strip().split(",")[0] for path in df['images'].tolist()]
    return findings, impressions, image_paths

# Load test and training data
test_findings, test_impressions, test_image_paths = load_data('/home/choonghan/practice/data_test_corrupted_0.5.csv')
train_findings, train_impressions, train_image_paths = load_data('/home/choonghan/practice/data_train_corrupted_0.5.csv')

# Initialize model, tokenizer, and device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1
)
checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)
model.to(device)

# Initialize BM25
train_tokenized_corpus = [finding.split() for finding in train_findings]
train_bm25 = BM25Okapi(train_tokenized_corpus)

# Load existing results or initialize a new dictionary
results_dict = {}
try:
    with open("/home/choonghan/practice/results_bm25_test_retriever_50_findings.json", "r") as f_json:
        results_dict = json.load(f_json)
except FileNotFoundError:
    pass

# Function to generate impressions
def generate_impression(i, max_tokens_findings=350, max_tokens_impressions=350):
    image_path = test_image_paths[i].replace("/home/choonghan/practice/", "")

    # Skip if the result already exists
    if image_path in results_dict:
        print(f"Skipping generation for {image_path} as it already exists in the JSON file.")
        return

    tokenized_query = test_findings[i].split()
    scores = train_bm25.get_scores(tokenized_query)
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    top_indices = [idx for idx in sorted_indices if idx != i][:2]

    prompt = 'Be a radiologist and write "impression", which is a summarization of findings of a radiology image. These are example impressions.'
    for idx in top_indices:
        prompt += create_prompt_entry(train_findings[idx], train_impressions[idx], tokenizer, max_tokens_findings, max_tokens_impressions)

    # Add the current test finding to the prompt
    prompt += create_prompt_entry(test_findings[i], "", tokenizer, max_tokens_findings, max_tokens_impressions, add_impression=False)

    # Process images
    images = [Image.open(path.strip()) for path in [train_image_paths[idx] for idx in top_indices] + [test_image_paths[i]]]
    vision_x = torch.cat([image_processor(img).unsqueeze(0) for img in images], dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device)

    # Tokenize the prompt for language model
    tokenizer.padding_side = "left"
    lang_x = tokenizer([prompt], return_tensors="pt", padding_side="left").to(device)

    # Generate the impression
    start_time = time.time()
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=20000,
        num_beams=3,
    )
    elapsed_time = time.time() - start_time
    result = tokenizer.decode(generated_text[0], skip_special_tokens=True)[len(prompt):].strip()

    print(f"Generated text: {result}")
    print(f"Time taken for generation: {elapsed_time:.2f} seconds")

    # Save the result in the dictionary
    results_dict[image_path] = result

    # Save the results to a JSON file
    with open("/home/choonghan/practice/results_bm25_test_retriever_50_findings.json", "w") as f_json:
        json.dump(results_dict, f_json, indent=4)

# Function to create prompt entries
def create_prompt_entry(finding, impression, tokenizer, max_tokens, max_tokens_impressions, add_impression=True):
    # Tokenize and truncate the finding
    tokens_findings = tokenizer.encode(finding.strip(), add_special_tokens=False)[:max_tokens]
    finding_text = tokenizer.decode(tokens_findings).strip()
    prompt_entry = f'<image>Findings: {finding_text}'
    
    if add_impression:
        # Tokenize and truncate the impression
        tokens_impressions = tokenizer.encode(impression.strip(), add_special_tokens=False)[:max_tokens_impressions]
        impression_text = tokenizer.decode(tokens_impressions).strip()
        prompt_entry += f'.Impression:{impression_text}'

    return prompt_entry

# Main loop to process each finding
for i in range(len(test_findings)):
    generate_impression(i)
