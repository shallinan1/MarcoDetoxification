# MaRCo Detoxification
This is the repository for the 2023 ACL Paper ["Detoxification with MaRCo: Controllable Revision with Experts and Anti-Experts"](https://arxiv.org/abs/2212.10543)

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

Given that the anti-expert is toxic and can be used for nefarious purposed by itself, access requests are currently enabled for this model. Requests will be automatically approved, but we have the right to revoke access if you misuse the model (spreading hate online, etc.).

If you want to train your own expert/anti-expert models on BART with custom hyperparemeter/dataset, please see the `training` folder of this repository.


## Citing this Work
If you use/reference this work, please cite us with:

    @article{Hallinan2022DetoxifyingTW,
      title={Detoxifying Text with MaRCo: Controllable Revision with Experts and Anti-Experts},
      author={Skyler Hallinan and Alisa Liu and Yejin Choi and Maarten Sap},
      journal={ArXiv},
      year={2022},
      volume={abs/2212.10543}
    }

## Contact

If you have any issues with the repository, questions about the paper, or anything else, please email hallisky@uw.edu.

