import os
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

from datasets import load_dataset

import torch

from template import (preprocess_slimorca_mistral, preprocess_alpaca_mistral, preprocess_orcamath_mistral,
                      preprocess_slimorca_gemma, preprocess_alpaca_gemma, preprocess_orcamath_gemma
)

from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse

tqdm.pandas()

def set_seed(seed_value=42):
    random.seed(seed_value) 
    np.random.seed(seed_value)  
    torch.manual_seed(seed_value)  
    torch.cuda.manual_seed_all(seed_value) 

def main(args):
    set_seed(42)

    model_name = args.model_path.split('/')[0]

    if model_name == 'mistralai':
        if args.data_name == "slimorca":
            dataset = load_dataset("Open-Orca/SlimOrca-Dedup", split='train')
            dataset = dataset.map(preprocess_slimorca_mistral)


        if args.data_name == "alpaca":
            dataset = load_dataset("tatsu-lab/alpaca", split='train')
            dataset = dataset.map(preprocess_alpaca_mistral)

        ## math
        if args.data_name == "orcamath":
            dataset = load_dataset("microsoft/orca-math-word-problems-200k", split='train')
            dataset = dataset.map(preprocess_orcamath_mistral)
            

    if model_name =='google':
        if args.data_name == "slimorca":
            dataset = load_dataset("Open-Orca/SlimOrca-Dedup", split='train')
            dataset = dataset.map(preprocess_slimorca_gemma)
            

        if args.data_name == "orcamath":
            dataset = load_dataset("microsoft/orca-math-word-problems-200k", split='train')
            dataset = dataset.map(preprocess_orcamath_gemma)
                  
        if args.data_name == "alpaca":
            dataset = load_dataset("tatsu-lab/alpaca", split='train')
            dataset = dataset.map(preprocess_alpaca_gemma)


    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    answer_start = "### Assistant:"
    special_tokens = {"additional_special_tokens": [answer_start]}
    tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.resize_token_embeddings(len(tokenizer))

    answer_ids = tokenizer(answer_start)['input_ids'][1]

    def calculate_token_length(text):
        return {'token_length': len(tokenizer.tokenize(text['text']))}
    
    dataset = dataset.map(calculate_token_length)
    dataset = dataset.filter(lambda example: example['token_length'] <= args.max_length)

    

    bar = tqdm(enumerate(dataset), total=len(dataset))

    def calc_data(text):
        inputs = tokenizer(text, return_tensors='pt')
        
        answer_start_idx = (inputs['input_ids'] == answer_ids).nonzero(as_tuple=True)[1]
        
        if answer_start_idx.nelement() != 0:
            answer_start_idx = answer_start_idx[0] + 1
            extracted_values = inputs['input_ids'][0, answer_start_idx:]
        else:
            extracted_values = torch.tensor([], device=model.device)

        labels_length = inputs['input_ids'].size(1)
        labels = torch.full((1, labels_length), -100, device=model.device)

        labels[0, :extracted_values.size(0)] = extracted_values

        inputs['labels'] = labels

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs)
            

        attentions = logits.attentions

        std_devs_per_layer = []
        for layer in range(len(attentions)):
            layer_attentions = attentions[layer]
            std_devs_per_token = []
            
            for token_idx in range(answer_start_idx, inputs['input_ids'].size(1)):
                attention_scores = layer_attentions[0, :, token_idx, :token_idx]
                std_dev = torch.std(attention_scores, dim=-1).mean().item()
                std_devs_per_token.append(std_dev)

            std_devs_per_layer.append(torch.mean(torch.tensor(std_devs_per_token)).item())

        mean_std_dev_across_layers = torch.mean(torch.tensor(std_devs_per_layer)).item()
            
        return logits.loss.item(), mean_std_dev_across_layers
    

    loss_list = []
    attention_list = []

    for i, data in bar:
        loss, attention = calc_data(data['text'])
        loss_list.append(loss)
        attention_list.append(attention)

    df = dataset.to_pandas()
    df['loss'] = loss_list
    df['attention'] = attention_list

    data_name = args.model_path.split('/')[0] +  '_' + args.data_name

    df.to_parquet(f'{data_name}.parquet', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--data_name', type=str, default='alpaca')
    parser.add_argument('--max_length', type=int, default=1024)

    args = parser.parse_args()
    main(args)