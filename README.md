# Multimodal, Efficient Radiology Report Summarization

Contributors: Choonghan Kim, Seonhee Cho, Jiho Lee, Chetan Chilkunda <br>
Collaborators: Joo Heung Yoon, M.D at the University of Pittsburgh Medical Center

<br>
This study explores robust multimodal strategies in the shared task of radiology report summarization.

Specifically, this work aims to quantify the effects of radiology image input on different model architectures in the context of 
increasingly corrupt text data. Baseline models include text-to-text ImpressionGPT and multimodal CheXOFA. In the setting of increasingly 
corrupt text input, we expect the multimodal models to perform better and we propose two new multimodal pipelines that leverage the image 
input to generalize against corrupt text input. The expectation is that these text-agnostic generalizations become part of the state-of-the-art 
pipelines for robust radiology report summarization. A secondary aspect of this work is the efficacy of leveraging prompt-based strategies and 
large language models over pre-training and fine-tuning approaches because of data privacy constraint in the medical domain.

This work was motivated by a course project (11-785 Introduction to Deep Learning) by Professor Bhiksha Raj. <br>
We are currently working towards publishing this work in Spring 2024.

The directory above details each of the three currently implemented models: <br>
NOTE: the files in these directories correspond to modified files from the main model directories on Github or Hugging Face
- CheXOFA with standard and Low Rank Adaptation (LoRA) fine-tuning methodologies
- Flamingo
- Instruct-BLIP with LLaMA-7b
