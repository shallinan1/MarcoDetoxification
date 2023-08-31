# Evaluation Metrics

## Toxicity
The file `toxicity.py` has the function `get_toxicity` to get PerspectiveAPI toxicity scores on one or more texts. To run this function, you will need to replace 14 in this file with your API key from Perspective API. You may also need to adjust the `qps` field, which designates the queries per second for Perspective API.

## Perplexity
Perplexity is done by default with a `gpt2-xl` model, but the model/tokenizer can be changed by passing in different parameters. By default, the `get_perplexity` method in `perplexity.py` omits the perplexity of sentences with values > 1e4 due to edge cases with `gpt2-xl`.

## BLEU, BertSscore

The implementations for these metrics are in `bleu.py` and `bertscore.py` respectively.

## Run all Metrics
Run the file `evaluate_all.py` with the following command, to get toxicity, BLEU, BERTScore, and perplexity for original texts and their rewrites with the following command, making sure you are in the main `MarcoDetoxification` directory:

    python3 -m evaluation.evaluate_all --orig_path your_original_text_path --gen_path your_rewrites_path

By default, the metrics will be saved to `your_rewrites_path[:-4] + "_stats.txt"`
