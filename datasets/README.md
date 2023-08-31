# Datasets

These are the datasets used for evaluation and training:

## Evaluation
### <ins> DynaBench </ins>

Contains the data from the full DynaHate paper (last round) and located under `dynabench` #TODO look into more details of this

### <ins> Microagressions.com</ins>

Contains data from [microaggressions.com](microaggressions.com), a Tumblr blog of self-reported microaggressions "in the wild". Located under `micoagressions`

### <ins> Social Bias Frames </ins>

Contains data from social bias frames (todo, link), specifically the microaggression portion. Located under `sbf`.

## Training

### <ins>Jigsaw</ins>

Jigsaw dataset, split into toxic and non-toxic subsets. Toxic subset consists of data where >0.5 proportion of annotators rated toxic, while non-toxic subset consists of data where 0 proportion of annotators rated toxic. Located under `jigsaw_full_30`. We host only the toxic portion of the dataset since the nontoxic portion is too large. Please download the dataset from (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)[https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge]

Jigsaw data has too columns: `masked` and `comment_text`. `comment_text` contains the text data from Jigsaw. The `masked` column is currently unused, but was a variant of `comment_text` where 30% of tokens were randomly masked. Instead, we now dynamically mask the tokens during training in `training/finetune_bart.py`.

## Model Outputs

#TODO add model outputs