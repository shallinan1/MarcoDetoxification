# Finetuning the toxic and nontoxic language models
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, AdamW, EarlyStoppingCallback
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import nn
import argparse
import random
from IPython import embed
from utils import *
from training.infilling import text_infill

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not torch.cuda.is_available():
        print("No GPUs found!")
    else:
        print("Found", str(torch.cuda.device_count()), "GPUS!")

    seed_everything(args.seed)

    # Load in the tokenizer
    tokenizer = BartTokenizer.from_pretrained(args.tok_type)
    mask = tokenizer.mask_token
    
    if not os.path.exists(args.model_dir):
        print(args.model_dir)
        os.mkdir(args.model_dir)

    output_dir = args.model_dir + "/" + args.model_type.split("/")[-1] + "_" + str(args.lr) + "_" + \
    str(args.seed) + "_" + str(args.train_batch_size * torch.cuda.device_count()) + "_" + args.data_type
    print(output_dir)

    # Logic to continue training - look at previous models saved
    try:
        prev_models = os.listdir(output_dir)
        # Alpha sort
        prev_models.sort()
        # Len sort
        prev_models.sort(key=len)
    except:
        prev_models = []
    
    # Logic to continue training if we want to load the old model - load pretrained model
    if args.load_old and len(prev_models) > 0:
        model = BartForConditionalGeneration.from_pretrained(os.path.join(output_dir, prev_models[-1]), forced_bos_token_id = tokenizer.bos_token_id).to(device)
    else:
        # Otherwise train a new model
        model = BartForConditionalGeneration.from_pretrained(args.model_type, forced_bos_token_id = tokenizer.bos_token_id).to(device)

    train_texts = []
    val_texts = []

    # Read/process the data based on which dataset we're using: Jigsaw or Dynabench
    # If you want to load your own data, put the data loading logic here
    if "jigsaw" in args.data_type:
        train = pd.read_csv(args.train_data)
        val = pd.read_csv(args.val_data)

        train_texts =  train["comment_text"].tolist()
        val_texts = val["comment_text"].tolist()
    elif "dynabench" in args.data_type:
        df = pd.read_csv(args.train_data)
        df_lab = "hate"
        if "nothate" in args.data_type:
            df_lab = "nothate"
        if "all" in args.data_type:
            df = df[df.label == df_lab]
        else:
            df_round = int(args.data_type[-1])
            inputs = df[df.label == df_lab][df["round.base"] == df_round]
        train_texts = df[df.split == "train"].text.tolist()
        val_texts = df[df.split == "dev"].text.tolist()

    print(len(train_texts), len(val_texts))

    # # Test percentiles of tokenized lengths
    # src_lengths = [len(tokenizer(x).input_ids) for x in train_texts]
    # # tgt_lengths = [len(tokenizer(x).input_ids) for x in train_labels]
    # print(np.percentile(src_lengths, 99))
    # embed()

    # Tokenize everything
    tokenized_labs_train = tokenizer.batch_encode_plus(
        train_texts, 
        max_length = args.max_target_length, 
        padding="max_length", 
        truncation=True, 
        return_tensors = "pt").input_ids

    tokenized_labs_val = tokenizer.batch_encode_plus(
        val_texts,
        max_length = args.max_target_length, 
        padding="max_length", 
        truncation=True, 
        return_tensors = "pt").input_ids
            
    tokenized_labs_val[tokenized_labs_val == tokenizer.pad_token_id] = -100
    tokenized_labs_train[tokenized_labs_train == tokenizer.pad_token_id] = -100

    # Convert this so that examples are dynamically masked
    class JigsawDataset(torch.utils.data.Dataset):
        def __init__(self, labels, rawdata):
            self.rawdata = rawdata
            self.labels = labels

        def __getitem__(self, idx):
            # Dynamically generate sources
            # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            # item['labels'] = cur_lab

            # Do detokenization to ensure that the tokenization matches up later
            item = tokenizer.encode_plus(
                detokenize(text_infill(self.rawdata[idx], tokenizer.mask_token)),
                max_length = args.max_source_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors = "pt")
            
            # Necessary for squeezing purposes
            item = {key: torch.tensor(val[0]) for key, val in item.items()}
            item['labels'] = self.labels[idx]
            return item
            
        def __len__(self):
            return len(self.labels)

    train_dataset = JigsawDataset(tokenized_labs_train, train_texts)
    val_dataset = JigsawDataset(tokenized_labs_val, val_texts)

    class BartTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(
                input_ids = inputs.get("input_ids"), 
                attention_mask = inputs.get("attention_mask"),
                labels = inputs.get("labels")
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return (loss, outputs) if return_outputs else loss

        def evaluate(self, **kwargs):
            metrics = super().evaluate(**kwargs)
            # Example to see what the model outputs for mask token (see if it outputs something biased)
            inputs_masked = ["Asians are <mask>"]
            print("\n Generations: ", tokenizer.batch_decode(model.generate(tokenizer(inputs_masked, return_tensors = "pt").to('cuda').input_ids, do_sample=True, num_return_sequences = 5)),"\n")                                                                                                                                                                        
            return metrics

    # embed()

    # Print out what the model looks like before we start training
    print("Before training:")
    inputs_masked = ["Asians are <mask>"]
    print("\n Generations: ", tokenizer.batch_decode(model.generate(tokenizer(inputs_masked, return_tensors = "pt").to('cuda').input_ids, do_sample=True, num_return_sequences = 5)),"\n")   

    training_args = TrainingArguments(
        output_dir=output_dir,          # output directory
        max_steps=args.max_steps,              # total number of training steps
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,   # batch size for evaluation
        learning_rate = args.lr,
        evaluation_strategy = "steps",
        save_strategy = "steps",
        save_steps = args.save_steps, 
        eval_steps = args.save_steps,
        fp16 = args.fp16,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        load_best_model_at_end = True,
        save_total_limit = args.save_total_limit,
        logging_dir=args.logging_dir,            # directory for storing logs
        logging_steps=args.logging_steps,
        seed = args.seed
    )

    trainer = BartTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        callbacks = [EarlyStoppingCallback(args.early_stopping_steps)]
    )

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok_type", default = "facebook/bart-base", help = "Tokenizer of model to fine-tune")
    parser.add_argument("--model_type", default = "facebook/bart-base", help = "Model to fine-tune")
    parser.add_argument("--train_data", default = "/gscratch/xlab/hallisky/rewriting/src/data/datasets/jigsaw_full_30/train_toxic.csv", help = "Path to train set; ether the toxic or nontoxic split of jigsaw")
    parser.add_argument("--val_data", default = "/gscratch/xlab/hallisky/rewriting/src/data/datasets/jigsaw_full_30/val_toxic.csv", help = "Path to dev set; same split as above")
    parser.add_argument("--model_dir", default = "/gscratch/xlab/hallisky/rewriting/models/toxic")
    parser.add_argument("--max_source_length", type = int, default = 182, help = "max source length (based on 99th percentile)")
    parser.add_argument("--max_target_length", type = int, default = 232, help = "max target length (based on 99th percentile)")
    parser.add_argument("--train_batch_size", type = int, default = 8)
    parser.add_argument("--eval_batch_size", type = int, default = 64)
    parser.add_argument("--max_steps", type = int, default = 50000)
    parser.add_argument("--lr", type = float, default = 2.5e-5)
    parser.add_argument("--logging_steps", type = int, default = 500)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--fp16", action = "store_true")
    parser.add_argument("--save_total_limit", type = int, default = 2)
    parser.add_argument("--save_steps", type = int, default = 500)
    parser.add_argument("--data_type",default = "jigsaw_full_30")
    parser.add_argument("--logging_dir", default = '/media/drive2/skyler/rewriting_data/toxic/models/logs')
    parser.add_argument("--early_stopping_steps", type = int, default = 5)
    parser.add_argument("--load_old", action="store_true", help="use if you want to continue training the previous model")
    main(parser.parse_args())
