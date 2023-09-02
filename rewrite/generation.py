import argparse
from pathlib import Path
from typing import Union, List
import os
from numpy import ufunc
from transformers import BartForConditionalGeneration, BartTokenizer
from IPython import embed
from infilling import *
from utils import *
import nltk.tokenize.casual
import torch
import torch.nn.functional as F
import sys
from . import gen_utils
from . import generation_logits_process
import pandas as pd
import functools
import operator
from .masking import method1
from tqdm import tqdm

"""
Infiller module
- Initialize with a base model, antiexpert (optional), expert (optional)
- If expert_type == "none", don't use an expert. Same for antiexpert
- 
"""
class Infiller():
    def __init__(
        self, 
        seed = 0, 
        base_path = "facebook/bart-base", 
        antiexpert_path  = "facebook/bart-base",
        expert_path = "facebook/bart-base",
        base_type = "base",
        antiexpert_type = "antiexpert",
        expert_type = "expert",
        tokenizer = "facebook/bart-base"
        ):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if not torch.cuda.is_available():
            print("No GPUs found!")
        else:
            print("Found", str(torch.cuda.device_count()), "GPUS!")

        self.seed = seed
        seed_everything(self.seed)

        # Initalize tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer)

        # Save mask info
        self.mask = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id

        model_map = {"base": base_path, "antiexpert": antiexpert_path, "expert": expert_path}

        # Initialize models
        self.base_model = None
        if base_type != "none":
            self.base_model = BartForConditionalGeneration.from_pretrained(model_map[base_type], forced_bos_token_id = self.tokenizer.bos_token_id).to(self.device)

        self.antiexpert = None
        if antiexpert_type != "none":
            self.antiexpert = BartForConditionalGeneration.from_pretrained(model_map[antiexpert_type], forced_bos_token_id = self.tokenizer.bos_token_id).to(self.device)

        self.expert = None
        if expert_type != "none":
            self.expert = BartForConditionalGeneration.from_pretrained(model_map[expert_type], forced_bos_token_id = self.tokenizer.bos_token_id).to(self.device)

    """
    Public generate model
    Parameters
    * inputs - list of text inputs
    * inputs_masked - list of text inputs with potentially toxic tokens masked
    * max_length - maximum length to generate too
    * sample - whether to sample or not
    * filter_p - nucleus sampling parameter on base logits
    * k - top_k parameter
    * p - nucleus sampling parameter on ensembled logits
    * temperature - for sampling
    * alpha_a - weight on antiexpert for ensenmbling distributions during decoding
    * alpha_e - weight on expert for ensenmbling distributions during decoding
    * alpha_b - weight on base model for ensenmbling distributions during decoding
    * repetition_penalty - how much to penalize repetition
    * batch_size - how many seqs to generate at once
    * verbose - whether or not to print generations during generation time
    """
    def generate(self,
            inputs: Union[str, List[str]],
            inputs_masked: Union[str, List[str]],
            max_length: int = 128,
            sample: bool = False,
            filter_p: float = 1.0,
            k: int = 0,
            p: float = 1.0,
            temperature: float = 1.0,
            alpha_a: float = 0.0,
            alpha_e: float = 0.0,
            alpha_b: float = 1.0,
            repetition_penalty: float = 1.0,
            batch_size = 50,
            verbose = False):
        
        # Initialize repetition penalty processor
        rep_penalty_proc = generation_logits_process.RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty) if repetition_penalty != 1.0 else None

        # Set models to eval
        if self.base_model:
            self.base_model.eval()
        if self.expert:
            self.expert.eval()
        if self.antiexpert:
            self.antiexpert.eval()

        final_outputs = []

        # Call private gen method either one time, or iterating through the inputs, batch_size at a time
        with torch.no_grad():
            if len(inputs) <= batch_size:
                outputs = gen(inputs = inputs, inputs_masked = inputs_masked, tokenizer = self.tokenizer, \
                    model = self.base_model, expert = self.expert, antiexpert = self.antiexpert, \
                    alpha_a = alpha_a, alpha_e = alpha_e, alpha_b = alpha_b, max_length = max_length, \
                    verbose=verbose, temperature = temperature, rep_proc = rep_penalty_proc, \
                    device = self.device, k = k, p = p, filter_p = filter_p, sample = sample)
                final_outputs = outputs
            else:
                max_seq_lens = 0
                for i in tqdm(range(0, len(inputs), batch_size), desc = "Filling in masks"):
                    cur_inputs = inputs[i:i+batch_size]
                    cur_inputs_masked = inputs_masked[i:i+batch_size]

                    outputs = gen(inputs = cur_inputs, inputs_masked = cur_inputs_masked, tokenizer = self.tokenizer, \
                        model = self.base_model, expert = self.expert, antiexpert = self.antiexpert, \
                        alpha_a = alpha_a, alpha_e = alpha_e, alpha_b = alpha_b, max_length = max_length, \
                        verbose=verbose, temperature = temperature, rep_proc = rep_penalty_proc, \
                        device = self.device, k = k, p = p, filter_p = filter_p, sample = sample)

                    max_seq_lens = max(max_seq_lens, outputs.shape[1])
                    final_outputs.append(outputs)
            
                final_outputs = [torch.nn.functional.pad(f, pad=(0, max_seq_lens - f.shape[1]), value=self.tokenizer.pad_token_id) \
                    for f in final_outputs]

                final_outputs = torch.cat(final_outputs)

        # Return both the tokenized outputs and the decoded outputs
        return final_outputs, self.tokenizer.batch_decode(final_outputs, skip_special_tokens = True)

# Private method for generation that is called by generate()
def gen(
    inputs, 
    inputs_masked, 
    tokenizer, 
    model, 
    expert, 
    antiexpert, 
    alpha_a: float = 0.0, 
    alpha_e: float = 0.0, 
    alpha_b: float = 1.0,
    max_length: int = 128, 
    device = torch.device('cuda'),
    verbose: bool = False, 
    sample: bool = False,
    filter_p: float = 1.0, 
    k: int = 0, 
    p: float = 1.0, 
    temperature: float = 1.0,
    rep_proc = None):

    # Convert inputs to list if they aren't already
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(inputs_masked, list):
        inputs = [inputs_masked]

    assert len(inputs) == len(inputs_masked)

    # Tokenize - the regular inputs, and the masked inputs
    batch = tokenizer(inputs, return_tensors='pt', padding = True).to(device)
    batch_masked = tokenizer(inputs_masked, return_tensors='pt', padding = True).to(device)

    if verbose:
        print("ORIGINAL \t"); print(tokenizer.batch_decode(batch.input_ids))
        print("\t"); print(tokenizer.batch_decode(batch_masked.input_ids))

    # Keep track of which generations aren't finished yet
    unfinished_sents = torch.ones(len(inputs), dtype=torch.int32, device=device)    

    # Start off our outputs with the eos token id, then the bos token id (match how BART generates)
    outputs = torch.Tensor([tokenizer.eos_token_id,tokenizer.bos_token_id]).expand(len(inputs), -1).long().to(device)
    start_length = 2

    loop_idx = 0
    # Substract start length from max length, since we start with 2 tokens
    while loop_idx < (max_length - start_length):

        # Compute the logits for base, antiexpert, and expert
        # Base model sees the nonmasked inputs, expert and antiexpert see the masked inputs
        base_logits = model.forward(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], decoder_input_ids = outputs).logits
        antiexpert_logits = antiexpert.forward(input_ids = batch_masked["input_ids"], attention_mask = batch_masked["attention_mask"], decoder_input_ids = outputs).logits
        expert_logits = expert.forward(input_ids = batch_masked["input_ids"], attention_mask = batch_masked["attention_mask"], decoder_input_ids = outputs).logits
        
        if verbose:
            print("Current outputs\n\t", tokenizer.batch_decode(outputs))
            print("Base\n")
            for idxs in torch.topk(base_logits[:,-1,:], 5, dim=-1).indices:
                print("\t", tokenizer.batch_decode(idxs))
            # print("Base masked", tokenizer.batch_decode(torch.topk(base_logits2[:,-1,:], 10).indices[0]))
            print("Anti\n")
            for idxs in torch.topk(antiexpert_logits[:,-1,:], 5, dim=-1).indices:
                print("\t", tokenizer.batch_decode(idxs))
            print("Expert\n")
            for idxs in torch.topk(expert_logits[:,-1,:], 5, dim=-1).indices:
                print("\t", tokenizer.batch_decode(idxs))
            # print("Expert nonmasked", tokenizer.batch_decode(torch.topk(expert_logits2[:,-1,:], 10).indices[0]))
        
        # eos_predicted = torch.argmax(base_logits[:,-1,:], dim=-1) == tokenizer.eos_token_id
        
        # top_p filtering on the base logits
        if filter_p < 1.0:
            base_logits = gen_utils.top_k_top_p_filtering(base_logits, top_p=filter_p)

        # Change values of the logits with the temperature
        # Temperature (higher temperature => more likely to sample low probability tokens)
        if temperature != 1.0:
            base_logits = base_logits / temperature

        # Ensemble the logits and get the next token logits
        ensemble_logits = alpha_b * base_logits + alpha_e * expert_logits - alpha_a * antiexpert_logits
        next_token_logits = ensemble_logits[:, -1, :]

        # Add repetition penalty
        if rep_proc is not None:
            next_token_logits = rep_proc(outputs, next_token_logits)
        
        # Sample or greedily decode from the next_token_logits
        if sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            # if temperature != 1.0:
            #     next_token_logits = next_token_logits / temperature
            if k > 0 or p < 1.0:
                next_token_logits = gen_utils.top_k_top_p_filtering(next_token_logits, top_k=k, top_p=p)
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        # Get the tokens to add and identify sentences that are done generating
        tokens_to_add = next_tokens * unfinished_sents + tokenizer.pad_token_id * (1 - unfinished_sents)
        eos_in_sents = tokens_to_add == tokenizer.eos_token_id
        unfinished_sents.mul_((~eos_in_sents).int())

        # Update the outputs and the loop index
        outputs = torch.cat((outputs, tokens_to_add.unsqueeze(-1)), dim=-1)
        loop_idx += 1

        if verbose:
            print("Ensemble\n")
            for idxs in torch.topk(ensemble_logits[:,-1,:], 5, dim=-1).indices:
                print("\t", tokenizer.batch_decode(idxs))
            print("Next token:", tokenizer.batch_decode(tokens_to_add))

        # Stop generation when there is an EOS in each sentence
        if unfinished_sents.max() == 0:
            break

    if verbose:
        decodes = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        print("MINE:")
        for d in decodes:
            print("\t", d)
        generated_ids = model.generate(batch_masked['input_ids'], max_length = max_length, num_beams = 1, do_sample = False)
        output = "\n\t".join(tokenizer.batch_decode(generated_ids, skip_special_tokens = True))
        print("INPUT\n\t", "\n\t".join(inputs)); print("\nbase OUTPUT\n\t", output)
        generated_ids = expert.generate(batch_masked['input_ids'], max_length = max_length, num_beams = 1, do_sample = False)
        output = "\n\t".join(tokenizer.batch_decode(generated_ids, skip_special_tokens = True))
        print("\nexpert OUTPUT\n\t", output)

    return outputs

if __name__ == '__main__':
    # Some examples of using the generate method on some masked inputs with alpha_a = 0.5, alpha_e = 2.5, and temperature = 1.25
    # If you want to run just the Infiller and generate method from the command line, you can modify the below to take in a list of masked inputs, process them, and feed them into the generate method
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha_a", type = float, default = 0.0)
    parser.add_argument("--alpha_e", type = float, default = 0.0)
    parser.add_argument("--temperature", type = float, default = 1.0)
    parser.add_argument("--alpha_b", type = float, default = 1.0)
    parser.add_argument("--max_length", type = int, default = 128)
    parser.add_argument("--verbose",action="store_true")

    rewriter = Infiller(
        seed = 0, 
        base_path = "facebook/bart-base", 
        antiexpert_path = "hallisky/bart-base-toxic-antiexpert",\
        expert_path = "hallisky/bart-base-nontoxic-expert", \
        base_type = "base", 
        antiexpert_type = "antiexpert", 
        expert_type = "expert", \
        tokenizer = "facebook/bart-base"
        )
    args = parser.parse_args()

    inputs =  ["I'm surprised you got it done, seeing as you're all girls!", "You are a human"]
    inputs_masked = ["<mask> surprised you got it done, seeing as you're all<mask>!", "You are a<mask>"]

    inputs = ["Person 1: I'm tired\nPerson 2: Nobody fucking cares bro", "Person 1: You are an idiot man\nPerson 2: Nah, you are the bitch", \
        "Person 1: In our society, there is a lot of wrongdoing, particularly by elites and those who are untouchable\nPerson 2: What the fuck are you talking about?"]
    inputs_masked = ["Person 1: I'm tired\nPerson 2: Nobody<mask> <mask> bro", "Person 1: You are an idiot man\nPerson 2: Nah, you<mask> the<mask>", \
        "Person 1: In our society, there is a lot of wrongdoing, particularly by elites and those who are untouchable\nPerson 2: What the<mask> are you<mask> about?"]
    context = ["I'm tired", "You are an idiot man","In our society, there is a lot of wrongdoing, particularly by elites and those who are untouchable"]
    
    rewriter.generate(inputs, inputs_masked, alpha_a = 0.5, alpha_e = 2.5, temperature = 1.25)
    #embed()