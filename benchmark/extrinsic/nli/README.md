# Extrinsic Benchmark: Natural Language Inference

## Main Results

In this task, we train on the [SNLI](https://nlp.stanford.edu/projects/snli/) dataset ([Bowman et al., 2015](https://nlp.stanford.edu/pubs/snli_paper.pdf)), and run inference on the Bias-NLI evaluation dataset from [Dev et al., 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6267). 

### Preprocessing

The SNLI data is automatically downloaded from HuggingFace during training, but you must manually process the evaluation data 
following instructions from the [code repo](https://github.com/sunipa/On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings) for "On Measuring and Mitigating Biased Inferences of Word Embeddings." Please ensure that the Bias-NLI data is formatted in the same way as the SNLI dataset from HuggingFace; that is, in a `.csv` file with the columns `[premise, hypothesis, label]` and stored under this directory as `bias-nli.csv`. 

Alternatively, you can also download our processed version of the Bias-NLI dataset from the Drive link [here](https://drive.google.com/file/d/1eC003yjOHjkp5-TGyVXW1emlV80qB7Yl/view?usp=sharing). 

### Training

The training arguments are stored in `run.sh`. At its present state, the script fine-tunes on SNLI data with MABEL and stores checkpoints in `nli-mabel`. 
We take the fine-tuned checkpoint that performs the best on the SNLI validation set for evaluation on Bias-NLI. 

```
chmod +x run.sh && ./run.sh
```

### Evaluation

```
python eval.py --load-from-file nli-mabel/checkpoint_best.pt --data-path bias-nli.csv
```

We provide the MABEL checkpoint from the Drive link [here](https://drive.google.com/file/d/1cOSnepKz0o_577oeYmgq8fgk5P9-83Vk/view?usp=sharing). Running evaluation gives the following numbers:

```
total net neutral:        0.9170128866319063
total fraction neutral:   0.9828041166841379
total tau 0.5 neutral:    0.9824839530908697
total tau 0.7 neutral:    0.9681385585408803
```


## Misc.

For the linear probing experiments in **Appendix I**, simply re-run the experiments with the `--fix-encoder` flag. 

