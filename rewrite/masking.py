import argparse
from pathlib import Path
from typing import Union, List
import os
from transformers import BartForConditionalGeneration, BartTokenizer
from IPython import embed
from infilling import *
from utils import preprocess, detokenize, seed_everything
import nltk.tokenize.casual
import torch
import torch.nn.functional as F
import sys
from . import gen_utils
from . import generation_logits_process
import pandas as pd
import functools
import operator
from tqdm import tqdm
import re
import html
import string

# Find needle in the haystack
def find_in_seq(haystack, needle):
    cands = [None]
    for i in range(len(haystack)):
        if torch.equal(haystack[i:i+len(needle)], needle):
            cands.append(i +len(needle))
    return cands[-1]

def find_in_seq_list(haystack, needle):
    cands = [None]
    for i in range(len(haystack)):
        if haystack[i:i+len(needle)] == needle:
            cands.append(i + len(needle))
    return cands[-1]

# Jensen divergence
def js_div(a,b, reduction):
    return 0.5 * F.kl_div(F.log_softmax(a, dim=-1), F.softmax(b,dim=-1), reduction=reduction) + \
         0.5 * F.kl_div(F.log_softmax(b, dim=-1), F.softmax(a,dim=-1), reduction=reduction) 

"""
Main Masker Class
- Initialized with seed, base model, antiexpert, expert, and tokenizer
- mask() method will apply MaRCO masking procedure to find where antiexpert/expert disagree and mask these locations
- Given a list of text in inputs, mask() returns the masked versions of these texts, where bad tokens are replaced with <mask> token (this is BART's mask token)
"""
class Masker():
    def __init__(
        self, 
        seed = 0, 
        base_path = "facebook/bart-base", 
        antiexpert_path  = "facebook/bart-base",
        expert_path = "facebook/bart-base",
        tokenizer = "facebook/bart-base"
        ):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if not torch.cuda.is_available():
            print("No GPUs found!")
        else:
            print("Found", str(torch.cuda.device_count()), "GPUS!")

        self.seed = seed
        seed_everything(self.seed)

        # Initalize self.tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer)

        # Initialize models
        self.model = BartForConditionalGeneration.from_pretrained(base_path, forced_bos_token_id = self.tokenizer.bos_token_id).to(self.device)
        self.antiexpert = BartForConditionalGeneration.from_pretrained(antiexpert_path, forced_bos_token_id = self.tokenizer.bos_token_id).to(self.device)
        self.expert = BartForConditionalGeneration.from_pretrained(expert_path, forced_bos_token_id = self.tokenizer.bos_token_id).to(self.device)
        self.model.eval()
        self.antiexpert.eval()
        self.expert.eval()
    
    """
    Takes in a list of text inputs, and a divergence threshold (thresh)
    Returns a list of the same text inputs, where some of the tokens are now replaced with <mask>
    """
    def mask(self,
        inputs,
        thresh = 1.5, # Divergence threshold to find which tokens to mask
        topk = 0, # Parameter not supported, can specify to mask the topk tokens with highest divergence rather than a random number of tokens above a threshold
        div_ba_thresh = 0.0, # Sets the threshold between base and other models if use_base_model_for_divergence=True. 
        use_base_model_for_divergence = False # If we want to use the base model's logits and compare to the expert/anti-expert. Purely for experimental purpose, this is NOT the implementation MaRCo uses and will likely perform worse. Set to False to match paper implementation.
    ):
        outputs = []

        batch = self.tokenizer(inputs, return_tensors='pt', padding = True).to(self.device)
        cur_labels = ["KL(base || exp)","KL(base || anti)","JS(exp || anti)"]
        for i in tqdm(range(len(inputs)), desc = "Identifying masks"):
            cur_tok = batch["input_ids"][i]    
            pad = torch.where(cur_tok == self.tokenizer.pad_token_id)[0]
            if len(pad) > 0:
                pad = pad[0]
                cur_tok = cur_tok[:pad]

            cur_seq = inputs[i]
            tok_map = {}
            casual = nltk.tokenize.casual.casual_tokenize(cur_seq)
            
            tok_map = {}
            old_idx = 1
            cur_idx = 0
            cur_word = casual[0]
            for new_idx, c in enumerate(cur_tok):
                d = self.tokenizer.decode(c).strip()
                if cur_word.startswith(d):
                    cur_word = cur_word.replace(d, "", 1)
                    if cur_word == "":
                        tok_map[cur_idx] = list(np.arange(old_idx, new_idx+1))
                        old_idx = new_idx+1
                        cur_idx += 1
                        try:
                            cur_word = casual[cur_idx]
                        except:
                            break 
            
            # Default MaRCO implementation: use only the expert and anti-expert and find divergence of prob. distributions on each token in the input
            if not use_base_model_for_divergence:
                # ignore start and end idxs
                ignore_idxs = []

                for c_idx, c in enumerate(casual):
                    punc_only = True
                    for k in c:
                        if k not in string.punctuation:
                            punc_only = False
                            break
                    if punc_only:
                        ignore_idxs.append(c_idx)

                sum_divs_ea = []
                for j in range(len(casual)):
                    new_seq = casual.copy()
                    new_seq[j] = self.tokenizer.mask_token
                    new_full_seq = detokenize(new_seq)
                    new_full_seq = re.sub(r"\s*<mask>", "<mask>", new_full_seq)

                    new_tok = self.tokenizer(new_full_seq,return_tensors="pt").input_ids.to(self.device)
                    mask_idx = torch.nonzero(new_tok[0] == self.tokenizer.mask_token_id)

                    expert_logits = self.expert.forward(input_ids = new_tok).logits
                    antiexpert_logits = self.antiexpert.forward(input_ids = new_tok).logits
                    divs_ea = js_div(expert_logits,antiexpert_logits, reduction='none').sum(dim = -1)
                    all_divs = []
                    for cor_idx in mask_idx:
                        all_divs.append(divs_ea[0][cor_idx.item()].item())
                    sum_divs_ea.append(np.mean(all_divs))

                # delete the ignore idxs
                mean_norm_ea = np.delete(sum_divs_ea, ignore_idxs)
                mean_norm_ea = np.array(mean_norm_ea) / mean_norm_ea.mean()
                above_thresh = np.nonzero(mean_norm_ea >= thresh)[0]

                new_casual=casual.copy()
                for a in above_thresh:
                    num_below = (np.array(ignore_idxs <= a)).sum()
                    new_casual[a + num_below] = self.tokenizer.mask_token

                outputs.append(re.sub(r"\s*<mask>", "<mask>",detokenize(new_casual)))

            else:
                # print(self.tokenizer.batch_decode(cur_tok), casual)
                base_logits = self.model.forward(input_ids = torch.unsqueeze(cur_tok, dim=0)).logits

                sum_divs_be, sum_divs_ba, sum_divs_ea = [],[],[]
                for j in range(len(cur_tok)):
                    new_tok = cur_tok.clone()
                    new_tok[j] = self.tokenizer.mask_token_id
                    expert_logits = self.expert.forward(input_ids = torch.unsqueeze(new_tok, dim=0)).logits
                    antiexpert_logits = self.antiexpert.forward(input_ids = torch.unsqueeze(new_tok, dim=0)).logits
                    
                    # Calculate the divergence between base and expert (be), base and antiexpert (ba), and expert and antiexpert (ea)
                    # If we want to look at js divergence instead of F.kl_div, use js_div

                    divs_be = F.kl_div(F.log_softmax(base_logits, dim=-1), F.softmax(expert_logits,dim=-1), reduction='none').sum(dim = -1)
                    divs_ba = F.kl_div(F.log_softmax(base_logits, dim=-1), F.softmax(antiexpert_logits,dim=-1), reduction='none').sum(dim = -1)
                    # divs_ea =F.kl_div(F.log_softmax(expert_logits, dim=-1), F.softmax(antiexpert_logits,dim=-1), reduction='none').sum(dim = -1)
                    divs_ea = js_div(expert_logits,antiexpert_logits, reduction='none').sum(dim = -1)

                    sum_divs_be.append(divs_be[0][j].item())
                    sum_divs_ba.append(divs_ba[0][j].item())
                    sum_divs_ea.append(divs_ea[0][j].item())
                
                cur_divs = [sum_divs_be, sum_divs_ba, sum_divs_ea]

                # Try rewriting here
                mean_norm_be = np.array(cur_divs[0]) / np.array(cur_divs[0]).mean()
                mean_norm_ba = np.array(cur_divs[1]) / np.array(cur_divs[1]).mean()
                mean_norm_ea = np.array(cur_divs[2]) / np.array(cur_divs[2]).mean()
                intersection = np.intersect1d(np.nonzero(mean_norm_ba >= div_ba_thresh)[0], np.nonzero(mean_norm_ea > thresh)[0])
                
                masked_idxs = cur_tok.clone()
                    
                for z in intersection:
                    for _, item in tok_map.items():
                        if z in item:
                            for it in item:
                                masked_idxs[it] = self.tokenizer.mask_token_id
                            break
                outputs.append(masked_idxs)

        if use_base_model_for_divergence:
            outputs = self.tokenizer.batch_decode(outputs,skip_special_tokens=False)
        return outputs

if __name__ == '__main__':
    # Below is a simple example using the Masker method and mask class on a couple of examples with a threshold of 1.25
    # If you want to run just the Masker from the command line, you can modify the below to take in a list of inputs, process them, and feed them into the mask method

    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh", type = float, default = 1.5, help = "Divergence threshold to identify which tokens to mask")

    # The following are too experimental parameters
    parser.add_argument("--topk", type = int, default = 0, help = "Parameter corresponding to method not implemented yet in mask method; choosing top k tokens with highest divergence")
    parser.add_argument("--div_ba_thresh", type = float, default = 0.0, help = "Divergence parameter if you want to use the base model in your divergence (NOT recommended)")

    masker = Masker(
        seed = 0, 
        base_path = "facebook/bart-base", 
        antiexpert_path = "hallisky/bart-base-toxic-antiexpert",\
        expert_path = "hallisky/bart-base-nontoxic-expert", \
        tokenizer = "facebook/bart-base"
        )
    args = parser.parse_args()

    inputs =  ["I'm surprised you got it done, seeing as you're all girls!", "You are a human"]
    inputs_masked = ["<mask> surprised you got it done, seeing as you're all<mask>!", "You are a<mask>"]
    
    masker.mask(inputs, thresh=1.25)
    embed()



