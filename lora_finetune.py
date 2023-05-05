import torch
import argparse
import torch.nn as nn
#import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import os
from peft import LoraConfig, get_peft_model 
from datasets import load_dataset
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer


class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    return None



def get_args_parser():
    parser = argparse.ArgumentParser('Finetune-LLM using peft adapters and LoRA', add_help=False)
    
    #train 
    parser.add_argument('--model', type=str, default="decapoda-research/llama-7b-hf", help='Model location (it works with Hugging--Face id)')
    parser.add_argument('--dataset', type=str, default="mrm8488/CHISTES_spanish_jokes", help='Dataset location (it works with Hugging--Face id)')
    parser.add_argument('--seed', type=int, default = 19920309, help='seed')
    return parser

def format_ds(example):
    example["text"] = "<SC>" + example['text'] + "<EC>"
    return example

def main (args):
    
    #load model and tokenizer 
    print ("::::INFO: LOADING MODEL:::::::")
    model = LlamaForCausalLM.from_pretrained(
                args.model, 
                #load_in_8bit=True, 
                device_map='auto',
                cache_dir=".hf_cache/"
            )
    
    tokenizer =  LlamaTokenizer.from_pretrained(args.model, cache_dir=".hf_cache/")
    #tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    new_tokens = ["<SC>","<EC>"]
    num_added_toks = tokenizer.add_tokens(new_tokens)
    print("::::INFO: We have added", num_added_toks, "tokens")
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    model.resize_token_embeddings(len(tokenizer))
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)
    #LoRA setup
    config = LoraConfig(
                       r=16,
                       lora_alpha=32,
                       target_modules=["q_proj", "v_proj"],
                       lora_dropout=0.05,
                       bias="none",
                       task_type="CAUSAL_LM"
                      )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)


    #load dataset 
    print ("::::INFO: LOADING DATASET:::::::")
    dataset = load_dataset(args.dataset,split='train')
    dataset = dataset.map(format_ds, remove_columns=['id', 'keywords', 'funny', 'category'])
    dataset = dataset.train_test_split(test_size=.05, seed=args.seed)

    train_dataset, val_dataset = dataset["train"], dataset["test"]
    
    print(dataset['train'][0])
    
    
    
    #tokenize samples
    train_dataset = train_dataset.map(lambda samples: tokenizer(samples['text']), batched=True)
    val_dataset = val_dataset.map(lambda samples: tokenizer(samples['text']), batched=True)
    #setup a trainer 
    import ipdb;ipdb.set_trace()
    trainer = transformers.Trainer(
    model=model, 
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=8, 
        gradient_accumulation_steps=8,
        warmup_steps=100, 
        max_steps=1000, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=5, 
        output_dir='train_adapters'
        ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    #trainer.train()
    trainer.train()
    import ipdb;ipdb.set_trace()
    model.push_to_hub("waybarrios/llama7b-es-finetuned-chistes_spanish_jokes-1000", use_auth_token=True)
    tokenizer.push_to_hub("waybarrios/llama7b-es-finetuned-chistes_spanish_jokes-1000", use_auth_token=True)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Finetune-LLM using peft adapters and LoRA', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)



