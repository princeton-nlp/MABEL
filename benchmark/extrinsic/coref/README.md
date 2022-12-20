# Extrinsic Benchmark: Coreference Resolution
## Data Pre-processing
To prepare the OntoNotes 5.0 dataset (Weischedel et al., 2013) dataset for training, you can run the following script:

```bash
model=... // encoder of choice
chmod +x setup_data.sh && ./setup_data.sh /path/to/ontonotes /path/to/processed/data model
```

## Training 

During training, we fine-tune a span-based, e2e coref model (Xu and Choi, 2020) using this [Pytorch implementation](https://github.com/lxucs/coref-hoi). We follow all default hyper-parameters and don't use any higher-order inference methods. 

Note: We use the **cased** version of all models—with uncased models, the fine-tuning performance deteriorates significantly. 
 
In `experiments.conf`, set `data_dir` appropriately. The training command is

```bash
python run.py train_mabel_bert_base [gpu_id]
```

Training takes <6 hours on a single NVIDIA V100 GPU.

## Evaluation 

We evaluate the fine-tuned coreference resolution models on the [WinoBias](https://uclanlp.github.io/corefBias/overview) benchmark (Zhao et al., 2018).  You can follow their processing instructions and have the data formatted as `.jsonlines`. Assuming the processed WinoBias files have been stored in the `/wb` folder, you can run this script to obtain the predictions: 

```bash
CONFIG=train_mabel_base
MODEL_CKPT=...
for PATH in type1_anti type1_pro type2_anti type2_pro;
do
  python predict.py --config_name=${CONFIG} --model_identifier=${MODEL_CKPT} --gpu_id=0 \
  --jsonlines_path=wb/${PATH}.jsonlines --output_path=scores/${CONFIG}-${PATH}.json
done
```
You can then run the commands from `evaluate_winobias.ipynb` to get the scores. Here are the numbers from the dev set for BERT and MABEL:

```
BERT:
T1A: 53.06
T1P: 86.21
T2A: 81.44
T2P: 93.41
TPR1: 33.15
TPR2: 11.97
```


```
MABEL:
T1A: 62.5
T1P: 83.19
T2A: 93.12
T2P: 95.38
TPR1: 20.69
TPR2: 2.26
```

### Collective Results

|              Models       | ON (F1) ↑ | 1A ↑ | 1P ↑ | 2A ↑| 2P ↑| TPR-1 ↓ | TPR-2 ↓ | 
|:-------------------------------|:------|:------|:------|:------|:------|:------|:------|
| [bert-base-uncased](https://drive.google.com/file/d/1mVs1wXjRVBbgDs7tQ30FaS-621CuziiV/view?usp=sharing) | 73.91 | 53.06 | 86.21 | 81.44 | 93.41 | 33.15 | 11.97 | 
| [princeton-nlp/mabel-bert-base-uncased](https://drive.google.com/file/d/1ySEFAYww7MfRA7s7bKLoZHzptzPdgKNe/view?usp=sharing) | 73.92 | 62.50 | 83.19 | 93.12 | 95.38 | 20.69 | 2.26 |
| [bert-large-uncased](https://drive.google.com/file/d/1f1m1oLw8FIDPf2zwTg0BXlYJBvMlV2hm/view?usp=sharing) | 77.39 | 66.44 | 91.36 | 93.69 | 98.85 | 24.92 | 5.16 | 
|  [princeton-nlp/mabel-bert-large-uncased](https://drive.google.com/file/d/1gMEa-79rdcKytTU-6Wrw5Hlf8KAPD30w/view?usp=sharing) | 77.00 | 70.37 | 85.22 | 96.47 | 98.85 | 14.84 | 2.38 | 

