NUM_GPU=1
export CUDA_VISIBLE_DEVICES=0,

CACHE_DIR=.cache

python train.py \
    --model_name_or_path princeton-nlp/mabel-bert-base-uncased \
    --cache_dir $CACHE_DIR \
    --ckpt_dir nli-mabel 
