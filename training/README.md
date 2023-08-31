Code to finetune the BART model is located in `finetune_bart.py`. 

To run, `cd` into `MarcoDetoxification/training` and use the command:

    python3 finetune_bart.py --args

Please see the file for the full list of arguments for training. You will need to train the model twice, one with toxic data to get the anti-expert, and one with nontoxic data to get the expert.