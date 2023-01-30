## MABEL: Attenuating Gender Bias using Textual Entailment Data

Authors: [Jacqueline He](https://jacqueline-he.github.io/), [Mengzhou Xia](https://xiamengzhou.github.io/), [Christiane Fellbaum](https://www.cs.princeton.edu/~fellbaum/), [Danqi Chen](https://www.cs.princeton.edu/~danqic/)

This repository contains the code for our EMNLP 2022 paper, ["MABEL: Attenuating Gender Bias using Textual Entailment Data"](https://arxiv.org/pdf/2210.14975.pdf). 

**MABEL** (a **M**ethod for **A**ttenuating **B**ias using **E**ntailment **L**abels) is a task-agnostic intermediate pre-training technique that leverages entailment pairs from NLI data to produce 
representations which are both semantically capable and fair. 
This approach exhibits a good fairness-performance tradeoff across intrinsic and extrinsic gender bias diagnostics, with minimal damage on natural language understanding tasks. 

![Training Schema](figure/teaser.png)


## Table of Contents
  * [Quick Start](#quick-start)
  * [Model List](#model-list)
  * [Training](#training)
  * [Evaluation](#evaluation)
    + [Intrinsic Metrics](#intrinsic-metrics)
    + [Extrinsic Metrics](#extrinsic-metrics)
    + [Language Understanding](#language-understanding)
  * [Code Acknowledgements](#code-acknowledgements)
  * [Citation](#citation)

## Quick Start

With the `transformers` package installed, you can import the off-the-shelf model like so: 

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/mabel-bert-base-uncased")

model = AutoModelForMaskedLM.from_pretrained("princeton-nlp/mabel-bert-base-uncased")
```

## Model List

|              MABEL Models       | ICAT ↑ |
|:-------------------------------|:------|
|  [princeton-nlp/mabel-bert-base-uncased](https://huggingface.co/princeton-nlp/mabel-bert-base-uncased) | 73.98 | 
| [princeton-nlp/mabel-bert-large-uncased](https://huggingface.co/princeton-nlp/mabel-bert-large-uncased) |  73.45 |
| [princeton-nlp/mabel-roberta-base](https://huggingface.co/princeton-nlp/mabel-roberta-base) |  69.68 |
| [princeton-nlp/mabel-roberta-large](https://huggingface.co/princeton-nlp/mabel-roberta-large) |  69.49 |

Note: The ICAT score is a bias metric that consolidates a model's capacity for language modeling and stereotypical association into a single numerical indicator. More information can be found in the [StereoSet](https://aclanthology.org/2021.acl-long.416.pdf) (Nadeem et al., 2021) paper.

## Training

Before training, make sure that the [counterfactually-augmented NLI data](https://drive.google.com/file/d/16KPp0rZv2DqAumccaRgvpLW4NwRCfdl6/view?usp=sharing), processed from SNLI and MNLI, is downloaded and stored under the `training` directory as `entailment_data.csv`. 

**1. Install package dependencies**

```bash
pip install -r requirements.txt
```

**2. Run training script**

```bash
cd training
chmod +x run.sh 
./run.sh
```
You can configure the hyper-parameters in `run.sh` accordingly. Models are saved to `out/`. The optimal set of hyper-parameters varies depending on the choice of backbone encoder, and the full training details can be found in the paper.   

## Evaluation

### Intrinsic Metrics 

If you use your own trained model instead of our provided HF checkpoint, you must first run `python -m training.convert_to_hf --path /path/to/your/checkpoint --base_model bert` (which converts the checkpoint to a standard BertForMaskedLM model - use `--base_model roberta` for RobertaForMaskedLM) prior to intrinsic evaluation.

Also, please note that we use [Meade et al.'s](https://arxiv.org/abs/2110.08527) method of computation and datasets for both StereoSet and CrowS-Pairs; this is why the metrics for the pre-trained models are not directly comparable to those reported in the original benchmark papers. 


**1. StereoSet ([Nadeem et al., 2021](https://aclanthology.org/2021.acl-long.416/))**

Command:

```bash
python -m benchmark.intrinsic.stereoset.predict --model_name_or_path princeton-nlp/mabel-bert-base-uncased && 
python -m benchmark.intrinsic.stereoset.eval
```


Output:
```
intrasentence
gender
Count: 2313.0
LM Score: 84.5453251710623
SS Score: 56.248299466465376
ICAT Score: 73.98003496789251
```

Collective Results:
|              Models       | LM ↑ | SS ◇| ICAT ↑ |
|:-------------------------------|:------|:------|:------|
| bert-base-uncased | 84.17 | 60.28 | 66.86 | 
|  princeton-nlp/mabel-bert-base-uncased | 84.54 | 56.25 | 73.98 | 
| bert-large-uncased | 86.54 | 63.24 | 63.62 | 
|  princeton-nlp/mabel-bert-large-uncased | 84.93 | 56.76 |  73.45 |
| roberta-base | 88.93 | 66.32 | 59.90 | 
|  princeton-nlp/mabel-roberta-base | 87.44 | 60.14 |  69.68 |
| roberta-large | 88.81 | 66.82 | 58.92 | 
|  princeton-nlp/mabel-roberta-large | 89.72 | 61.28 |  69.49 |


◇: The closer to 50, the better.

**2. CrowS-Pairs ([Nangia et al., 2021](https://aclanthology.org/2020.emnlp-main.154/))** 

Command: 

```bash
python -m benchmark.intrinsic.crows.eval --model_name_or_path princeton-nlp/mabel-bert-base-uncased
```


Output: 
``` 
====================================================================================================
Total examples: 262
Metric score: 50.76
Stereotype score: 51.57
Anti-stereotype score: 49.51
Num. neutral: 0.0
====================================================================================================
```

Collective Results:
|              Models       | Metric Score ◇ |
|:-------------------------------|:------|
| bert-base-uncased | 57.25 |
|  princeton-nlp/mabel-bert-base-uncased |50.76 | 
| bert-large-uncased | 55.73 |
|  princeton-nlp/mabel-bert-large-uncased | 51.15 |
| roberta-base | 60.15 | 
|  princeton-nlp/mabel-roberta-base | 49.04 | 
| roberta-large | 60.15 | 
|  princeton-nlp/mabel-roberta-large | 54.41 | 

◇: The closer to 50, the better.

### Extrinsic Metrics

1. Occupation Classification 

See `benchmark/extrinsic/occ_cls/README.md` for full training instructions and results.

2. Natural Language Inference

See `benchmark/extrinsic/nli/README.md` for full training instructions and results.

3. Coreference Resolution

See `benchmark/extrinsic/coref/README.md` for full training instructions and results.


### Language Understanding 
**1. GLUE ([Wang et al., 2018](https://aclanthology.org/W18-5446/))** 

We fine-tune on GLUE through the [transformers](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) library, following the default hyper-parameters. 

A straightforward way is to download the current transformers repository:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

Then set up the environment dependencies:

```bash
cd ./examples/pytorch/text-classification
pip install -r requirements.txt
```


Here is a sample script for one of the GLUE tasks, MRPC:

```bash
# task options: cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte 
export TASK_NAME=mrpc
export OUTPUT_DIR=out/

CUDA_VISIBLE_DEVICES=0 python run_glue.py \
  --model_name_or_path princeton-nlp/mabel-bert-base-uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir $OUTPUT_DIR
```


**2. SentEval Transfer Tasks ([Conneau et al., 2018](https://arxiv.org/abs/1803.05449))**

Preprocess:

Make sure you have cloned the [SentEval](https://github.com/facebookresearch/SentEval) repo and added its contents into this repository's `transfer` folder, and run `./get_transfer_data.bash` in `data/downstream` to download the evaluation data.

Command:

```bash
python -m benchmark.transfer.eval --model_name_or_path princeton-nlp/mabel-bert-base-uncased --task_set transfer
```

Output:

```
+-------+-------+-------+-------+-------+-------+-------+-------+
|   MR  |   CR  |  SUBJ |  MPQA |  SST2 |  TREC |  MRPC |  Avg. |
+-------+-------+-------+-------+-------+-------+-------+-------+
| 78.33 | 85.83 | 93.78 | 89.13 | 85.50 | 85.20 | 68.87 | 83.81 |
+-------+-------+-------+-------+-------+-------+-------+-------+
```

Collective Results:
|              Models       | Transfer Avg. ↑ |
|:-------------------------------|:------|
| bert-base-uncased | 83.73 |
|  princeton-nlp/mabel-bert-base-uncased |83.81 | 
| bert-large-uncased | 86.54 |
|  princeton-nlp/mabel-bert-large-uncased | 86.09 |


## Code Acknowledgements
- Evaluation code for StereoSet and CrowS-Pairs is adapted from ["An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models"](https://aclanthology.org/2022.acl-long.132/) (Meade et al., 2022). 
- Model implementation code is adapted from [SimCSE](https://github.com/princeton-nlp/SimCSE/blob/main/evaluation.py) (Gao et al., 2021). 
- Evaluation code for the transfer tasks relies on the SentEval package [here](https://github.com/facebookresearch/SentEval), and adapts from a script prepared by [SimCSE](https://github.com/princeton-nlp/SimCSE/blob/main/evaluation.py) (Gao et al., 2021). 
- Evaluation code for GLUE relies on the Huggingface implementation of the [transformers](https://arxiv.org/abs/1910.03771) (Wolf et al., 2019) package.
- Training and evaluation for e2e span-based coreference resolution follows from [this Pytorch implementation](https://aclanthology.org/2020.emnlp-main.686/) (Xu and Choi, 2020).
- Repository is formatted with [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black).

## Citation
```bibtex
@inproceedings{he2022mabel,
   title={{MABEL}: Attenuating Gender Bias using Textual Entailment Data},
   author={He, Jacqueline and Xia, Mengzhou and Fellbaum, Christiane and Chen, Danqi},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2022}
}
```
