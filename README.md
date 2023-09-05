# MaRCo Detoxification
This is the repository for the 2023 ACL Paper ["Detoxifying Text with MaRCo: Controllable Revision with Experts and Anti-Experts"](https://arxiv.org/abs/2212.10543)

<p align="center">
  <img src="https://pbs.twimg.com/media/FkeZuLBUUAA6abX?format=jpg&name=4096x4096" alt="drawing" width="75%"/>
</p>

## Using this Repository

### <ins>Setting up the Environment</ins>
To set up the environment to run the code, make sure to have conda installed, then run

    conda env create -f environment.yml

Then, activate the environment

    conda activate rewrite

### <ins>Expert Models</ins>

The expert and anti-expert models are available on huggingface here:
* [Expert](https://huggingface.co/hallisky/bart-base-nontoxic-expert): BART-base further finetuned on the ***non-toxic*** portion of the Jigsaw Corpus with the same pretraining masked denoising objective
* [Anti-Expert](https://huggingface.co/hallisky/bart-base-toxic-antiexpert): BART-base further finetuned on the ***toxic*** portion of the Jigsaw Corpus with the same pretraining masked denoising objective

You can download then load them or use these directly from the huggingface transfomers library, ie, 

    # Load model directly
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("hallisky/bart-base-nontoxic-expert")
    model = AutoModelForSeq2SeqLM.from_pretrained("hallisky/bart-base-nontoxic-expert")

If you want to train your own expert/anti-expert models on BART with custom hyperparameter/dataset, please see the `training` folder of this repository.

### <ins>Detoxification with MaRCo</ins>

See `rewrite/README.md` for details on how to run the detoxification pipeline,

### <ins>Datasets</ins>

See `datasets/README.md` for access to the datasets and a description. 

### <ins>Evaluation</ins>

See `evaluation/README.md` for the evaluation pipeline on detoxifications. 

## Citing this Work
If you use/reference this work, please cite us with:

    @inproceedings{hallinan-etal-2023-detoxifying,
        title = "Detoxifying Text with {M}a{RC}o: Controllable Revision with Experts and Anti-Experts",
        author = "Hallinan, Skyler  and
          Liu, Alisa  and
          Choi, Yejin  and
          Sap, Maarten",
        booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.acl-short.21",
        doi = "10.18653/v1/2023.acl-short.21",
        pages = "228--242",
        abstract = "Text detoxification has the potential to mitigate the harms of toxicity by rephrasing text to remove offensive meaning, but subtle toxicity remains challenging to tackle. We introduce MaRCo, a detoxification algorithm that combines controllable generation and text rewriting methods using a Product of Experts with autoencoder language models (LMs). MaRCo uses likelihoods under a non-toxic LM (expert) and a toxic LM (anti-expert) to find candidate words to mask and potentially replace. We evaluate our method on several subtle toxicity and microaggressions datasets, and show that it not only outperforms baselines on automatic metrics, but MaRCo{'}s rewrites are preferred 2.1 times more in human evaluation. Its applicability to instances of subtle toxicity is especially promising, demonstrating a path forward for addressing increasingly elusive online hate.",
    }


## Contact

If you have any issues with the repository, questions about the paper, or anything else, please email hallisky@uw.edu.

