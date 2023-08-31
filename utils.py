from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from sacremoses import MosesDetokenizer
import numpy as np
import torch
import random
import html
import re
import ftfy
from nltk.tokenize.casual import casual_tokenize

nl_tok = "[<NEW>]"
md = MosesDetokenizer(lang='en')

def detokenize(input):
    # return TreebankWordDetokenizer().detokenize(input)
    return md.detokenize(input)

def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def bool2str(cand):
    if cand:
        return "T"
    return "F"

def seed_everything(seed = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Only useful for convolution
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def preprocess(text, preserve_lines = False):
    if preserve_lines:
        return ftfy.fix_text(html.unescape(text))
    # Remove linee break and excess spaces
    return ftfy.fix_text(html.unescape(re.sub(r'\s+', ' ', text).strip()))

# Quick test
# TreebankWordDetokenizer.detokenize(TreebankWordTokenizer.tokenize("sh*t"))