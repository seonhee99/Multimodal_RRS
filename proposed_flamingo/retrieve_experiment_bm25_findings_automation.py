from open_flamingo import create_model_and_transforms
import torch
from huggingface_hub import hf_hub_download

from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random
from rank_bm25 import BM25Okapi
import time
import pandas as pd
random.seed(0)
# 파일에서 데이터를 읽어옵니다
import json
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


##############################################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
)
model.to(device)
checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)

# 훈련 데이터 로딩
df_train = pd.read_csv('/home/choonghan/practice/data/data_train_full.csv')
train_findings = df_train['findings'].tolist()
train_impressions = df_train['impressions'].tolist()
train_image_paths = ["/home/choonghan/practice/" + str(path).strip().split(",")[0] for path in df_train['images'].tolist()]
train_tokenized_corpus = [finding.split() for finding in train_findings]
train_bm25 = BM25Okapi(train_tokenized_corpus)

# 실험 데이터셋을 위한 for 루프
for test_type in ['corrupted_0.5', 'corrupted_0.8']:  # ['full', 'corrupted_0.2', 'corrupted_0.5', 'corrupted_0.8']
    df = pd.read_csv(f'/home/choonghan/practice/data/data_test_{test_type}.csv')
    findings = df['findings'].tolist()
    impressions = df['impressions'].tolist()
    image_paths = ["/home/choonghan/practice/" + path.strip().split(",")[0] for path in df['images'].tolist()]
    # 특정 인덱스만 사용하는 경우
    test_list = [  0,   1,   2,   3,   4,   5,   6,   7,   8,  11,  13,  14,  20,
             21,  22,  24,  25,  26,  27,  28,  29,  30,  31,  32,  34,  35,
             37,  38,  39,  40,  41,  42,  44,  45,  46,  47,  49,  50,  51,
             52,  53,  55,  56,  58,  59,  60,  61,  62,  63,  64,  66,  67,
             69,  70,  71,  72,  73,  74,  75,  76,  77,  79,  80,  82,  83,
             85,  86,  87,  88,  89,  91,  92,  93,  94,  97,  98, 100, 102,
            103, 104, 105, 106, 107, 111, 113, 115, 116, 117, 118, 119, 122,
            123, 124, 126, 128, 130, 132, 133, 134, 136]  # 전체 인덱스 목록
    findings = [findings[i] for i in test_list]
    impressions = [impressions[i] for i in test_list]
    image_paths = [image_paths[i] for i in test_list]

    tokenized_corpus = [finding.split() for finding in findings]
    try:
        with open(f"/home/choonghan/practice/result/results_bm25_test_retriever_3b_only_100_{test_type}.json", "r") as f_json:
            results_dict = json.load(f_json)
    except FileNotFoundError:
        results_dict = {}

    # 메인 실험 루프
    for i in range(0,len(findings)):
        image_path = image_paths[i].replace("/home/choonghan/practice/", "")

        # Check if the result for this image path already exists
        if image_path in results_dict:
            print(f"Skipping generation for {image_path} as it already exists in the JSON file.")
            continue

        try:
            tokenized_query = tokenized_corpus[i]

        
            
            # Get the scores for all findings
            scores = train_bm25.get_scores(tokenized_query)

            # Sort scores and get the indices of the top 2 documents (other than the query itself)
            sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            top_indices = [idx for idx in sorted_indices if idx != i][:2]

            # random_indices = top_indices
            
            max_tokens_findings = 350  # findings의 최대 토큰 수
            max_tokens_impressions = 350 

            # 랜덤으로 선택한 인덱스에 해당하는 image_paths
            selected_image_paths = [train_image_paths[idx] for idx in top_indices]
            # 랜덤으로 선택한 샘플의 image, findings, impressions를 프롬프트에 추가합니다.
            prompt = 'Be a radiologist and write "impression", which is a summarization of findings of a radiology image.These are example impressions.'
            for idx in top_indices:
                retrieved_finding = train_findings[idx].strip()
                tokens_findings = tokenizer.encode(retrieved_finding, add_special_tokens=False)
                # 최대 길이로 자르기
                if len(tokens_findings) > max_tokens_findings:
                    tokens_findings = tokens_findings[:max_tokens_findings]
                # 텍스트로 변환
                retrieved_finding = tokenizer.decode(tokens_findings).strip()


                retrieved_impression = train_impressions[idx].strip()
                tokens_impressions = tokenizer.encode(retrieved_impression, add_special_tokens=False)
                # 최대 길이로 자르기
                if len(tokens_impressions) > max_tokens_impressions:
                    tokens_impressions = tokens_impressions[:max_tokens_impressions]
                # 텍스트로 변환
                retrieved_impression = tokenizer.decode(tokens_impressions).strip()
                prompt += f'<image>Findings: {retrieved_finding}.Impression:{retrieved_impression}<|endofchunk|>'

            # 3번째 샘플의 image와 findings만 추가합니다.
            finding_str = findings[i].strip()
            # 3번째 샘플의 image와 findings만 추가합니다.
            tokens_findings = tokenizer.encode(finding_str, add_special_tokens=False)
            # 최대 길이로 자르기
            if len(tokens_findings) > max_tokens_findings:
                tokens_findings = tokens_findings[:max_tokens_findings]
                finding_str = tokenizer.decode(tokens_findings).strip()   
            prompt += f'<image>Findings: {finding_str}.Impression:'

            # print(prompt)
            a=3

            # 첫 두개의 findings와 그에 대응하는 이미지 로딩
            images = [Image.open(path.strip()) for path in selected_image_paths]
            images.append(Image.open(image_paths[i].strip()))
            # 이미지 처리
            vision_x = [image_processor(img).unsqueeze(0) for img in images]
            vision_x = torch.cat(vision_x, dim=0)
            vision_x = vision_x.unsqueeze(1).unsqueeze(0)

            # 프롬프트 작성
            tokenizer.padding_side = "left" # For generation padding tokens should be on the left
            lang_x = tokenizer(
                [prompt],
                return_tensors="pt",
            )
            # print(len(lang_x['input_ids'][0]))
            vision_x = vision_x.to(device)
            lang_x = {k: v.to(device) for k, v in lang_x.items()}
            start_time = time.time()
            generated_text = model.generate(
                vision_x=vision_x,
                lang_x=lang_x["input_ids"],
                attention_mask=lang_x["attention_mask"],
                max_new_tokens=20000,
                num_beams=3,
            )
            
            elapsed_time = time.time() - start_time
            print("Generated text: ", tokenizer.decode(generated_text[0]))
            print(f"Time taken for generation {elapsed_time:.2f} seconds")
            # result = tokenizer.decode(generated_text[0])[len(prompt):].strip()
            # print("Generated text: ", result)
            result = tokenizer.decode(generated_text[0])[len(prompt):].replace('<|endofchunk|>', '').strip()
            # print('proprecessed Generated text: ', result)
            print(f"Result {i + 1}: , {result}")

            # results.append((result))
            image_paths_new = image_paths[i].replace("/home/choonghan/practice/","")
            reference = impressions[i]
            results_dict[image_path] = {
                'prompt': prompt,
                'gold': reference,
                'pred': result,
                'time': elapsed_time
            }
            # results_dict[image_paths_new] = result

        except AssertionError as error:
            print(f"Skipping index {i} due to error: {error}")
            continue 

        with open(f"/home/choonghan/practice/result/results_bm25_test_retriever_3b_only_100_{test_type}.json", "w") as f_json:
            json.dump(results_dict, f_json, indent=4)

