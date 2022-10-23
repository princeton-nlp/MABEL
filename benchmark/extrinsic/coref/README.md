# Extrinsic Benchmark: Coreference Resolution
## Training 

During training, we fine-tune the encoder on the OntoNotes 5.0 dataset (Weischedel et al., 2013) using a [Pytorch implementation](https://github.com/lxucs/coref-hoi) of a span-based, e2e coref model (Xu and Choi, 2020). We follow their default hyper-parameters and don't use any higher-order inference methods. 

Note: We use the **cased** version of all models - with uncased models, the fine-tuned performance deteriorates significantly. 
 
In the `experiments.conf` section, set `data_dir` appropriately. The training command is

```
python run.py train_mabel [gpu_id]
```

Training takes <6 hours on a single NVIDIA V100 GPU.

## Evaluation 

We evaluate the fine-tuned coreference resolution models on the [WinoBias](https://uclanlp.github.io/corefBias/overview) benchmark (Zhao et al., 2018).  You can follow their processing instructions and have the data formatted as `.jsonlines`. Assuming the processed WinoBias files have been stored in the `/wb` folder, you can run this script to obtain the predictions: 

```
CONFIG=mabel
MODEL_CKPT=...
for PATH in type1_anti type1_pro type2_anti type2_pro;
do
  python predict.py --config_name=train_${CONFIG}_base --model_identifier=${MODEL_CKPT} --gpu_id=0 \
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

The model checkpoints that produce these numbers can be found [here](https://drive.google.com/file/d/1mVs1wXjRVBbgDs7tQ30FaS-621CuziiV/view?usp=sharing) for BERT, and [here](https://drive.google.com/file/d/1ySEFAYww7MfRA7s7bKLoZHzptzPdgKNe/view?usp=sharing) for MABEL.
