import random
import numpy as np
import torch
import argparse

from tqdm import tqdm

from trainer import CustomTrainer
from peft import (
    LoraConfig,
    get_peft_model,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
import wandb
import pandas as pd

tqdm.pandas()

wandb.login()
wandb.init(project="curriculum-learning-preorder")

def set_seed(seed_value=42):
    random.seed(seed_value)  # Python 내장 random 모듈의 시드 고정
    np.random.seed(seed_value)  # Numpy 모듈의 시드 고정
    torch.manual_seed(seed_value)  # CPU 연산을 위한 PyTorch 시드 고정
    torch.cuda.manual_seed_all(seed_value)  # 모든 GPU를 위한 PyTorch 시드 고정, GPU가 있을 경우


def main(args):
    set_seed(42)

    model_name = args.model_path.split('/')[0]
    data_path = args.data_path

    wandb.run.name = f'{model_name}_{args.data_name}_{args.order_type}'
    wandb.run.save()

    if model_name == 'mistralai':
        if args.data_name == "slimorca":
            df = pd.read_parquet(f'{data_path}')
            
        if args.data_name == "alpaca":
            df = pd.read_parquet(f'{data_path}')

        if args.data_name == "orcamath":
            df = pd.read_parquet(f'{data_path}')


    if model_name =='google':
        if args.data_name == "slimorca":
            df = pd.read_parquet(f'{data_path}')
            
        if args.data_name == "orcamath":
            df = pd.read_parquet(f'{data_path}')
                                
        if args.data_name == "alpaca":
            df = pd.read_parquet(f'{data_path}')

        


    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16


    print("df length: ", len(df))

    if args.order_type == "length":
        df = df.sort_values(by='token_length', ascending=True)

    if args.order_type == "loss":
        df = df.sort_values(by='loss', ascending=True)

    if args.order_type == "attention":
        df = df.sort_values(by='attention', ascending=False)

    train_ds = Dataset.from_pandas(df)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # 역양자화
    )

    model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            output_attentions=True,
            quantization_config=bnb_config,
            device_map="auto",
            use_cache=False
            )    

    if model_name == 'mistralai':
        target_module = ["q_proj","k_proj","v_proj","o_proj", 'gate_proj', 'up_proj', 'down_proj']

        model.config.sliding_window = None
        model.config.rope_theta = 1000000

    if model_name == 'google':
        target_module = ["q_proj","k_proj","v_proj","o_proj", 'gate_proj', 'up_proj', 'down_proj']
    
    lora_config = LoraConfig(
        r=64, # Lora attention dimension. 
        lora_alpha=16, # the alpha parameter for Lora scaling.
        lora_dropout=0.1, # the dropout probability for Lora layers.
        bias="none",
        target_modules=target_module,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    collator = DataCollatorForCompletionOnlyLM('### Assistant:', tokenizer=tokenizer)
    

    training_args = TrainingArguments(
                        output_dir=args.save_path, 
                        overwrite_output_dir=True,
                        save_total_limit=args.epochs,
                        evaluation_strategy="no",
                        max_grad_norm=0.3,
                        logging_strategy='epoch',
                        # logging_strategy='steps',
                        # logging_steps='100',
                        per_device_train_batch_size=args.batch_size, # 8
                        num_train_epochs=args.epochs,
                        lr_scheduler_type="constant",
                        learning_rate=args.lr,
                        save_strategy='epoch',
                        optim="paged_adamw_32bit", # 연산 방식
                        bf16=True, # mixed precision (32bit > 16bit만 사용해서 연산)
                        run_name=f"{model_name}_{args.data_name}_{args.order_type}",
                        report_to="wandb",
                        do_eval=False,
                    )
     
    dq_trainer = CustomTrainer(
                            model,
                            train_dataset=train_ds,
                            args=training_args,
                            tokenizer=tokenizer,
                            dataset_text_field="text",
                            max_seq_length=args.max_length,
                            data_collator=collator,
                            order_type=args.order_type # options = loss, attention
                        )
    
    dq_trainer.train()



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--data_name', type=str, default='alpaca')
    parser.add_argument('--data_path', type=str, default='./mistralai_orcamath.csv')
    parser.add_argument('--lr', type=float, default=2e-4)

    parser.add_argument('--max_length', type=int, default=1024)
    
    parser.add_argument('--save_path', type=str, default='./sft')
    parser.add_argument('--order_type', type=str, default='random')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    main(args)
