from sacrebleu.metrics import BLEU
from IPython import embed
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

def get_bleu(references,hypotheses):
    bleu = BLEU()
    return bleu.corpus_score(hypotheses, references).score

def calc_bleu(inputs, preds):
    bleu_sim = 0
    counter = 0
    for i in tqdm(range(len(inputs))):
        if len(inputs[i]) > 3 and len(preds[i]) > 3:
            bleu_sim += sentence_bleu([inputs[i]], preds[i])
            counter += 1
    return float(bleu_sim / counter)

if __name__ == "__main__":
    # Some testing
    references = [["I am going to the store today to buy milk"], ["The dog bit his shoe, and he screamed"]]
    hypotheses = ["Going to store is something I will do today", "The angry wolf tore at his shoe. He screamed!"]
    bleu_score = get_bleu(references, hypotheses)


