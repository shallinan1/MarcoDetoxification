import argparse
from pathlib import Path
from typing import Union, List
import os
os.environ['TRANSFORMERS_CACHE'] = 'cache/'
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
from tqdm import tqdm
from .masking import Masker, method1, method1_list, preprocess
from .generation import Infiller
import re
import time
import itertools

"""
Method to read in data from one of the datasets we ran experiments on
Additional code needed if you want to add your own datasets here
- Returns a list of texts to rewrite
"""
def get_data(args):
    inputs =  ["I'm surprised you got it done, seeing as you're all girls!", "You are a human", "You are a genius"]
    if args.data_path is not None:
        if "dynabench" in args.data_path:
            df = pd.read_csv(args.data_path)
            df_lab = "hate"
            df_split = "dev"

            if "test" in args.data_type:
                df_split = "test"
            if "train" in args.data_type:
                df_split = "train"
            if "nothate" in args.data_type:
                df_lab = "nothate"
            if "all" in args.data_type:
                inputs = df[(df.split == df_split) & (df.label == df_lab)].text.tolist()
            else:
                df_round = int(args.data_type[-1])
                inputs = df[(df.split == df_split) & (df.label == df_lab)][df["round.base"] == df_round].text.tolist()

        elif "sbf" in args.data_path:
            df = pd.read_csv(args.data_path)
            dataSource = "redditMicroagressions"
            
            if "nonoff" in args.data_type:
                inputs = df[df.dataSource ==dataSource ][df.offensiveYN < 0.5].post.tolist()
            else:
                inputs = df[df.dataSource ==dataSource ][df.offensiveYN >= 0.5].post.tolist()


        elif "microagressions" in args.data_path:
            df = pd.read_csv(args.data_path)
            inputs = [preprocess(s) for s in df.actual_quote.tolist()]

    return inputs

"""
Sample method to 1) load data in 2) mask the inputs and save to a file and 3) infill the masked inputs using specified generation parameters
"""
def rewrite(args):
    # Get the inputs to rewrite
    inputs = get_data(args)

    # Specifying the path to save the maksed inputs to. Feel free to change this based on your file name
    mask_path = "masked_thresh" + str(args.thresh)
    cur_path = os.path.join(args.output_dir, args.data_type, mask_path)
    print(cur_path)

    # Check if we already have the masked inputs if we want to reuse previously masked inputs; args.overwrite_mask means we will regenerate the masked inputs regardless
    # Branch: generate new masked versions of the inputs. Feel free to replace this logic 
    if not os.path.exists(os.path.join(cur_path, "masked_inputs.txt")) or args.overwrite_mask:       
        try:
            os.makedirs(cur_path)
        except:
            pass

        # Initilaize the Makser object with the parameters from the args
        masker = Masker(
            seed = args.seed, base_path = args.base_path, antiexpert_path = args.antiexpert_path,\
            expert_path = args.expert_path, tokenizer =  args.tokenizer
        )

        # Use the mask function from makser to mask the inputs with a specified threshold
        decoded_masked_inputs = masker.mask(inputs=inputs, thresh=args.thresh)

        # Hacky way to remove bos and eos token from decoded mask inputs and save to text file
        decoded_mask_inputs = [d.replace("<s>", "").replace("</s>", "") for d in decoded_masked_inputs]
        with open(os.path.join(cur_path, "masked_inputs.txt"), "w") as f:
            for d in decoded_mask_inputs:
                f.write(re.sub(r"\s+", " ", d) + "\n") 
    # Branch: Reused previously masked inputs instead of generating them again
    else:
        try: 
            # Load the previously generated inputs
            with open(os.path.join(cur_path, "masked_inputs.txt"), "r") as f:
                decoded_mask_inputs = [s.strip() for s in f.readlines()]
            print("Skipping masking step by using old masks")
        except:
            # Exit if we aren't able to load inputs properly
            print("No file found. Exiting")
            import sys; sys.exit()

    # Initialize our Infiller class
    rewriter = Infiller(
        seed = args.seed, base_path = args.base_path, antiexpert_path = args.antiexpert_path,\
        expert_path = args.expert_path, base_type = args.base_type, antiexpert_type = args.antiexpert_type, \
        expert_type = args.expert_type, tokenizer = args.tokenizer
    )

    # We have a parameter "gen_many" that describes if we want to make many generations with different hyperparamaters with the same masked inputs. The default is NOT to do this, and only make one generation for the set of inputs
    if not args.gen_many:

        # The filename to save the generations under. Feel free to use your own logic here
        gen_path = "aa" + str(args.alpha_a) + "_ae" + str(args.alpha_e) + "_ab" + str(args.alpha_b) + "_base" + args.base_type[:5] + \
            "_anti" + args.antiexpert_type[:5] + "_expert" + args.expert_type[:5] + "_temp" + str(args.temperature)  + \
            "_sample" + bool2str(args.sample)  + "_topk" + str(args.top_k_gen) + "_reppenalty" + str(args.rep_penalty) + \
            "_filterp" + str(args.filter_p)  + "_maxlength" + str(args.max_length) + "_topp" + str(args.top_p) 
        final_path = os.path.join(cur_path, gen_path)
        print(final_path)
        
        # Check if we have already generated this file; args.overwrite_gen means we will regenerate regardless
        # Branch: Generate from the masked inputs
        if not os.path.exists(os.path.join(final_path, "gen.txt")) or args.overwrite_gen:
            try:
                os.mkdir(final_path)
            except:
                pass

            # Call the generate method
            outputs, decoded_outputs = rewriter.generate(inputs, decoded_mask_inputs, alpha_a = args.alpha_a, alpha_e = args.alpha_e, alpha_b = args.alpha_b, \
                temperature = args.temperature, verbose = args.verbose, max_length = args.max_length, repetition_penalty= args.rep_penalty, \
                top_p = args.top_p, filter_p = args.filter_p, top_k = args.top_k_gen, batch_size = args.batch_size, sample = args.sample)

            # Save the original inputs and the generations
            with open(os.path.join(final_path, "orig.txt"), "w") as f:
                for l in inputs:
                    f.write(re.sub(r"\s+", " ", l).strip() + "\n")
            with open(os.path.join(final_path, "gen.txt"), "w") as f:
                for l in decoded_outputs:
                    f.write(re.sub(r"\s+", " ", l).strip() + "\n")

            print(time.strftime('%l:%M%p %Z on %b %d, %Y').strip())
            print("Finished generation\n")
        else:
            # Exit, since we already have this gen
            print(time.strftime('%l:%M%p %Z on %b %d, %Y').strip())
            print("Exiting, since generation already exists\n")
            import sys; sys.exit()
    
    # We have a parameter "gen_many" that describes if we want to make many generations at once. The default is NOT to do this, and only make one generation for the set of inputs
    else:
        # Some of the parameters we want to iterate through
        alpha_as = np.arange(0.0, 2.1, 0.5)
        alpha_es = np.arange(2.0, 5.1, 0.25)
        temps = [0.9, 1.3, 1.7, 2.1, 2.5, 2.9]
        rep_p = [1.0, 1.2, 1.5]
        alpha_bs = [1.0]
        print("starting combos")
        import time
        t = 1000 * time.time() # current time in milliseconds
        np.random.seed(int(t) % 2**32)

        # Randomize the combinations that we iterate through
        all_combos = np.random.permutation(list(itertools.product(alpha_as, alpha_es, temps, alpha_bs,rep_p)))
        np.random.seed(0)

        # Go through the combos and make generations and save them, as in the above branch
        for combo in tqdm(all_combos):
            args.alpha_a = combo[0]
            args.alpha_e = combo[1]
            args.temperature = combo[2]
            args.alpha_b = combo[3]
            args.rep_penalty = combo[4]
            if args.alpha_a >= (args.alpha_e - 0.9):
                continue
            gen_path = "aa" + str(args.alpha_a) + "_ae" + str(args.alpha_e) + "_ab" + str(args.alpha_b) + "_base" + args.base_type[:5] + \
                "_anti" + args.antiexpert_type[:5] + "_expert" + args.expert_type[:5] + "_temp" + str(args.temperature)  + \
                "_sample" + bool2str(args.sample)  + "_topk" + str(args.top_k_gen) + "_reppenalty" + str(args.rep_penalty) + \
                "_filterp" + str(args.filter_p)  + "_maxlength" + str(args.max_length) + "_topp" + str(args.top_p)

            final_path = os.path.join(cur_path, gen_path)
            print(final_path)
            if not os.path.exists(os.path.join(final_path, "gen.txt")) or args.overwrite_gen:
                try:
                    os.mkdir(final_path)
                except:
                    pass

                outputs, decoded_outputs = rewriter.generate(inputs, decoded_mask_inputs, alpha_a = args.alpha_a, alpha_e = args.alpha_e, alpha_b = args.alpha_b, \
                    temperature = args.temperature, verbose = args.verbose, max_length = args.max_length, repetition_penalty = args.rep_penalty, \
                    top_p = args.top_p, filter_p = args.filter_p, top_k = args.top_k_gen, batch_size = args.batch_size, sample = args.sample)

                with open(os.path.join(final_path, "orig.txt"), "w") as f:
                    for l in inputs:
                        f.write(re.sub(r"\s+", " ", l).strip() + "\n")

                with open(os.path.join(final_path, "gen.txt"), "w") as f:
                    for l in decoded_outputs:
                        f.write(re.sub(r"\s+", " ", l).strip() + "\n")
                        
                print(time.strftime('%l:%M%p %Z on %b %d, %Y').strip())
                print("Finished generation\n")
            else:
                # Exit, since we already have this gen
                print(time.strftime('%l:%M%p %Z on %b %d, %Y').strip())
                print("\nskipping\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument("--seed", type = int, default = 0, help = "Random seed to use")
    parser.add_argument("--data_type", default ="manual")
    parser.add_argument("--data_path", default =None, help = "Path to the data")
    parser.add_argument("--output_dir", default = "data/dexp_outputs/", help = "Directory to save the outputs to")
    parser.add_argument("--base_path", default = "facebook/bart-base", help = "Base model to use")
    parser.add_argument("--antiexpert_path", default = "hallisky/bart-base-toxic-antiexpert", help = "Antiexpert model to use")
    parser.add_argument("--expert_path", default = "hallisky/bart-base-nontoxic-expert", help = "Expert model to use")
    parser.add_argument("--tokenizer", default = "facebook/bart-base", help = "Tokenizer to use")
    parser.add_argument("--base_type", default = "base", choices=["base", "expert", "antiexpert"], help = "Which model to for the base model")
    parser.add_argument("--expert_type", default = "expert", choices=["base", "expert", "antiexpert"], help = "Which model to use as the expert")
    parser.add_argument("--antiexpert_type", default = "antiexpert", choices=["base", "expert", "antiexpert"], help = "Which model to use for the antiexpert")
        
    # Args for Masking portion
    parser.add_argument("--thresh", type = float, default = 1.5, help = "Divergence threshold to identify which tokens to mask")
    parser.add_argument("--overwrite_mask",action="store_true", help = "If you want to regenerate the mask file for a set of inputs. Only useful if making multiple generations on the same input with different hyperparameters")

    # Args for Infilling portion (generation)
    parser.add_argument("--alpha_a", type = float, default = 0.0, help = "weight on antiexpert for ensenmbling distributions during decoding")
    parser.add_argument("--alpha_e", type = float, default = 0.0, help = "weight on expert for ensenmbling distributions during decoding")
    parser.add_argument("--alpha_b", type = float, default = 1.0, help = "weight on base model for ensenmbling distributions during decoding")
    parser.add_argument("--max_length", type = int, default = 128, help = "maximum length to generate too")
    parser.add_argument("--verbose", action="store_true", help = "whether or not to print generations during generation time")
    parser.add_argument("--temperature", type = float, default = 1.0, help = "generation temperature")
    parser.add_argument("--top_k_gen", type = int, default = 50, help = "top_k parameter for generation")
    parser.add_argument("--rep_penalty", type = float, default = 1.0, help = "how much to penalize repetition")
    parser.add_argument("--top_p", type = float, default = 1.0, help = "nucleus sampling parameter on ensembled logits")
    parser.add_argument("--filter_p", type = float, default = 1.0, help = "nucleus sampling parameter on base logits")
    parser.add_argument("--batch_size", type = int, default = 25, help = "how many seqs to generate at once")
    parser.add_argument("--sample", action="store_true", help = "whether to sample or not")
    parser.add_argument("--overwrite_gen",action="store_true", help = "If you want to regenerate the generation for a set of masked inputs. Only useful if making multiple generations on the same input with different hyperparameters")

    parser.add_argument("--gen_many", action="store_true", help = "Whether or not we want to generate multiple generations for a set of inputs with a lot of hyperparameters. We recommend setting this to False. If true, need to define your own hypers to iterate through (not in argparser)")

    args = parser.parse_args()
    rewrite(args)