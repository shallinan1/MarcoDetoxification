# https://huggingface.co/transformers/perplexity.html
import os
os.environ['TRANSFORMERS_CACHE'] = "/gscratch/xlab/hallisky/KAug/cache"
import transformers
import torch
from IPython import embed
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math
from tqdm import tqdm

def get_perplexity(sentences, model = None, tokenizer = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None or model is None:
        model_id = 'gpt2-xl'
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    model.eval()
    perp = []
    for sentence in tqdm(sentences):
        tokenized = tokenizer(sentence, return_tensors = "pt").to(device)
        with torch.no_grad():
            output = model(tokenized["input_ids"], labels=tokenized["input_ids"])
        
        final_perp = math.exp(output.loss.item())
        if final_perp < 1e4:
            perp.append(final_perp)
    return perp

if __name__ == "__main__":
    print(get_perplexity(["Hey man! How are you doing? It's been a really long time hasn't it", "gihasb?? d as  s s a  "]))
    # temp = get_perplexity(["Hey, that's not fair", "gihasb"])

