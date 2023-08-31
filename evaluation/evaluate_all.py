from .bertscore import *
from .bleu import *
from .perplexity import *
from .toxicity import *
import numpy as np
import argparse
import os

def evaluate_toxicity(references, hypotheses):
    print("Calculating toxicity...")
    tox_hyp = get_toxicity(hypotheses)
    percent_tox_hyp = sum(np.array(tox_hyp) >= 0.5) / len(tox_hyp)
    tox_hyp = np.nanmean(tox_hyp)

    return tox_hyp, percent_tox_hyp

def save_toxicity(tox_hyp, percent_tox_hyp, path, file_name = "toxicity.txt"):    
    with open(os.path.join(path, file_name), "w") as f:
        f.write("toxicity gen" + ": " + str(tox_hyp) + "\n")
        f.write("percent toxic gen" + ": " + str(percent_tox_hyp))

def evaluate_all(references, hypotheses, eval_ref=True):
    tox_ref, perp_ref, percent_tox_ref = None, None, None
    
    # print("Calculating toxicity...")
    # tox_hyp = get_toxicity(hypotheses)
    # percent_tox_hyp = sum(np.array(tox_hyp) >= 0.5) / len(tox_hyp)
    # tox_hyp = np.nanmean(tox_hyp)

    print("Calculating perplexity...")
    perp_hyp =  np.nanmean(get_perplexity(hypotheses))

    embed()
    print("Calculating bert score...")
    bs = np.nanmean(get_bert_scores(zip(references,hypotheses))["f1"])
    print("Calculating bleu4...")
    bleu = calc_bleu(references, hypotheses)

    if eval_ref:
        print("Calculating toxicity...")
        tox_ref = get_toxicity(references)
        percent_tox_ref = sum(np.array(tox_ref) >= 0.5) / len(tox_ref)
        tox_ref =  np.nanmean(tox_ref)
        print("Calculating perplexity...")
        perp_ref =  np.nanmean(get_perplexity(references))

    return bs, bleu, tox_hyp, perp_hyp, tox_ref, perp_ref, percent_tox_hyp, percent_tox_ref

def get_data(args):
    print(args)
    # Loading data from path
    with open(args.orig_path, "r") as f:
        orig = [s.strip() for s in f.readlines()] 
    
    with open(args.gen_path, "r") as f:
        gen = [s.strip() for s in f.readlines()] 

    return orig, gen

def eval_args(args):
    orig, gen = get_data(args)

    metrics = evaluate_all(orig, gen)
    name = args.gen_path.split('/')[-1]
    save_path = args.gen_path[:-4] + "_stats.txt"

    items = ["bertscore", "bleu4", "toxicity gen", "perplexity gen", "toxicity orig", "perplexity orig", "percent toxic gen", "percent toxic ref"]
    
    with open(save_path, "w") as f:
        for i, m in zip(items, metrics):
            print(i, ":", m)
            f.write(i + ": " + str(m) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_path")
    parser.add_argument("--gen_path")
    eval_args(parser.parse_args())
    
    """
    Some example commands

    python3 -m evaluation.evaluate_all --orig_path /home/skylerh/rewriting/src/data/test_10k_toxic.txt --gen_path /home/skylerh/rewriting/src/data/model_outputs/paragedi_with_mined_paraphraser.txt

    python3 -m evaluation.evaluate_all --orig_path /home/skylerh/rewriting/src/data/dexp_outputs/m-expert_data-jigsaw_mask_bathresh-0.5_eathresh-2.0_topk-0_alpha-0.0_orig.txt \
    --gen_path /home/skylerh/rewriting/src/data/dexp_outputs/m-expert_data-jigsaw_mask_bathresh-0.5_eathresh-2.0_topk-0_alpha-0.0_gen.txt

    python3 -m evaluation.evaluate_all --orig_path /home/skylerh/rewriting/src/data/test_10k_toxic.txt --gen_path /home/skylerh/rewriting/src/data/dexp_outputs/m-expert_data-jigsaw_mask_bathresh-0.5_eathresh-1.5_topk-0_alpha-0.0_orig.txt

    python3 -m evaluation.evaluate_all --gen_path /gscratch/xlab/hallisky/rewriting/src/data/dexp_outputs/base_expert_jigsaw_jigsaw_mask_bathresh0.5_eathresh1.5_topk0_alpha0.0_beams1/gen.txt --orig_path /gscratch/xlab/hallisky/rewriting/src/data/dexp_outputs/base_expert_jigsaw_jigsaw_mask_bathresh0.5_eathresh1.5_topk0_alpha0.0_beams1/orig.txt
        
    """
