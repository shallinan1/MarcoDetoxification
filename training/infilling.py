# Code for infilling and adding masks to sequence (used to randomly mask sequences in BART training)

import numpy as np
import nltk.tokenize.casual
import bisect
from IPython import embed

np.random.seed(0)

def max_span(num_mask, tokenized_len, thresh):
    i = 1
    while True:
        if (num_mask + i)/(tokenized_len) >= thresh:
            return i
        i += 1

def list_diffs(arr, max_len):
    if len(arr) == 0:
        return max_len
    else:
        # print(np.diff(np.array([0] + arr + [max_len - 1]) - 2))
        return np.max(np.diff(np.array([0] + arr + [max_len - 1]) - 2))

def collapse_contig(arr, token):
    output = []
    seen_prev = False
    for i in arr:
        if i == token:
            if seen_prev:
                seen_prev = True
                continue
            seen_prev = True
        else:
            seen_prev = False
        output.append(i)
    return output

def text_infill(sentence, mask_token, lam = 3, thresh = 0.3):
    tokenized = np.array(nltk.tokenize.casual.casual_tokenize(sentence), dtype = "object")
    masked_idcs = []

    while (len(masked_idcs) / len(tokenized)) < thresh:
        span_length = np.random.poisson(lam = lam)
        
        while ((span_length > list_diffs(masked_idcs, len(tokenized))) or \
            (span_length > max_span(len(masked_idcs), len(tokenized), thresh))):    
            span_length = np.random.poisson(lam = lam)
            # print("Span length is too long, it is currently:", span_length)

        # print("tokenized is", tokenized)
        # print("masked idcs are", masked_idcs)
        # print("span length is", span_length)

        if span_length == 0:
            start_idx = np.random.randint(0, len(tokenized) + 1)
            while ((start_idx in masked_idcs) or (start_idx in (np.array(masked_idcs) + 1))):
                # print("bad, start_idx is", start_idx)
                start_idx = np.random.randint(0, len(tokenized) + 1)
            
            # print("start idx is", start_idx)
            tokenized = np.insert(tokenized, start_idx, mask_token)
            bisect.insort(masked_idcs, start_idx)
        
        else:
            while True:          
                start_idx = np.random.randint(0, len(tokenized) - span_length + 1)
                idcs = np.arange(start_idx, start_idx + span_length)
                
                for i in idcs:
                    if i in masked_idcs or i in (np.array(masked_idcs) + 1):
                        # print("bad i" , i)
                        continue
                break
            
            for i in idcs:
                bisect.insort(masked_idcs, i)
                tokenized[i] = mask_token
            #print("idcs are", idcs)
    # print("final mask ratio:",len(masked_idcs)/len(tokenized))
    return collapse_contig(tokenized, mask_token)

# Masks tokens from idx to idx + span_length with mask_token
# If idx > length of sequence, does not mask anything
def span_mask(sentence, idx, mask_token, span_length = 1):
    tokenized = np.array(nltk.tokenize.casual.casual_tokenize(sentence), dtype = "object")
    max_len = len(tokenized)
    if idx < max_len:
        end_span = min(idx+span_length, max_len)
        tokenized[idx:end_span] = [mask_token for i in range(end_span - idx)]
    return tokenized

# embed()
# print(text_infill("I'm gonna go, do you want anything Mom?", "<mask>"))
# print(text_infill("Hey I'm going to the store, do you want anything?", "<mask>"))