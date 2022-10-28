# Extrinsic Benchmark: Occupation Classification
## Main Results
### Pre-processing

Please follow the data pre-processing instructions from the `biosbias` [code repo](https://github.com/microsoft/biosbias), from [de-Arteaga et al., 2019](https://dl.acm.org/doi/10.1145/3287560.3287572). By the end, you should have obtained a `BIOS.pkl` file, separated into `trainbios.pkl`, `devbios.pkl`, and `testbios.pkl` in a 65/25/10 split, and stored in this directory. 

Note that this dataset originally had 397,340 biographies; when we downloaded the dataset in Jan. 2022, it only had 206,511 biographies due to broken links.

### Training 

Run this command to fine-tune MABEL: 
`python train.py --model_name_or_path princeton-nlp/mabel-bert-base-uncased --ckpt_dir bios-mabel`  

MABEL will periodically evaluate on the validation set, and save the checkpoint with the highest total accuracy into the `bios-mabel` directory. 

### Evaluation

Run this command to evaluate MABEL:

```
best=bios-mabel/best_checkpoint.pt // replace with best ckpt path
python eval.py --load_from_file ${best}
```

You can find our fine-tuned checkpoint for MABEL [here](https://drive.google.com/file/d/1MP-8sAf3YJk279Dj1S54QYBtaQDgB4kc/view?usp=sharing). Evaluating this checkpoint will produce the following:

```
Bias-in-Bios evaluation results:
 - model checkpoint: bios-mabel/model-step=5000-acc=84.33.pt
 - acc. (all): 84.42877073321966
 - acc. (m): 84.70782959124928
 - acc. (f): 84.10312395028552
 - tpr gap: 0.6047056409637652
 - tpr rms: 0.1340815229504565
```

## Misc.
For the linear probing experiments in **Appendix I**, simply re-run the experiments with the `--fix-encoder` flag. 

