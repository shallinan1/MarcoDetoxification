# Using MaRCo for Detoxification

Given a text, the main components of MaRCo are 1) determining potentially toxic tokens in the text and masking them 2) infilling the masked tokens. To this end, we have the following code structure:

## Masking
`masking.py` contains code to mask inputs by comparing token probabilities under the expert and antiexpert models. Specifically, one must instantiate the `Masker()` class with the tokenizer path, seed, base model path, expert model path, and an antiexpert model path. The specific configuration we use recommend is:

    masker = Masker(
        seed = 0,
        base_path = "facebook/bart-base", 
        antiexpert_path  = "hallisky/bart-base-toxic-antiexpert",
        expert_path = "hallisky/bart-base-toxic-antiexpert",
        tokenizer = "facebook/bart-base"
    )

Then, using this `masker`, call the `mask()` method, which takes in a list of text inputs, and a divergence threshold (we use 1.2 in the paper. A lower divergence threshold means more tokens will be maksed, so this might be useful if toxicity reduction is especially important over meaning preservation; the opposite holds for a higher threshold). The method returns a list of the same text inputs, where potentially toxic tokens are replaced with `<mask>`. There are three other parameters detailed in the code for the `mask` method, but these are not recommended. The way to use the `mask` method is:

    decoded_masked_inputs = masker.mask(inputs, thresh = 1.2)

We also recommend removing the BOS and EOS tokens afterwards, ie, 

    decoded_mask_inputs = [d.replace("<s>", "").replace("</s>", "") for d in decoded_masked_inputs]

## Generating
`generation.py` contains code to infill the masked inputs from the Masking step using a product-of-experts with the BART models. Again, one must instantiate a class, this time being the `Infiller()` class. Please see the code for a description of all the initialization parameters. The specific configuration we recommend is:
    
    infiller = Infiller(
        seed = 0, 
        base_path = "facebook/bart-base", 
        antiexpert_path  = "hallisky/bart-base-toxic-antiexpert",
        expert_path = "hallisky/bart-base-toxic-antiexpert",
        base_type = "base",
        antiexpert_type = "antiexpert",
        expert_type = "expert",
        tokenizer = "facebook/bart-base"
    )

Then, using this `infiller` class, call the `generate()` method, which takes a list of the original inputs and the corresponding *masked* text inputs (you can pass in `inputs` and `decoded_mask_inputs` from above), and different generation parameters which are described directly above the method, such as temperature, the weight of the experts for product-of-experts decoding, etc. `generate()` will return the tokenized outputs and the decoded outputs. As an example, the generate call that we used for the microaggresions dataset (greedy decoding setup) looks like:

    outputs, decoded_outputs = generate(
            inputs
            decoded_mask_inputs,
            max_length = 128,
            sample = False,
            filter_p =  1.0,
            k = 0,
            p = 1.0,
            temperature = 2.5,
            alpha_a: float = 1.5,
            alpha_e: float = 4.25,
            alpha_b: float = 1.0,
            repetition_penalty: 1.0,
            batch_size = 50,
            verbose = False)

Feel free to change your generation parameters to support other decoding strategies (nucleus sampling, top-k, etc.)

## Example Script - Entire Pipeline of Masking and Generating
We include an example of the entire pipeline of masking and rewriting inputs in `rewrite.py`. Briefly, we have a method `get_data()` that will load one of the evaluation datasets we used in the paper (dynabench, sbf, or microagressions) given a data path, and use these as inputs. Then, the Masker is initialized and masks the inputs. Finally, the Infiller is initialized and the detoxified output is returned; the original inputs and the rewrites are then saved to a text file. 

The example script has logic to name files to save them, such as the masked inputs, and the detoxified texts; feel free to replace this with your own naming schema. In addition, the script has an argument `gen_many` that we recommend you only to use in a special case: if you want to try detoxifiyng the same text with many different generation hyperparameters. In the script, this parameter is set-up so that multiple sets of hyperparameters can be iterated through to make multiple potential rewrites.

Feel free to modify this file to accomodate your own data and/or other masking/generation strategies. The **argparser** is expansive, has desciprtions, and should cover all the parameters need to do data loading, masking, and then generation.


## Auxillary Files
`gen_utils.py` and `generation_logits_process.py` are helper functions for generation.
