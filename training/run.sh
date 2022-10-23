NUM_GPU=1

PORT_ID=$(expr $RANDOM + 1000)

export OMP_NUM_THREADS=8

## hyperparams ##
MAX_SEQ=128
DATA_PATH='entailment_data.csv'
CACHE_DIR='.cache'
BS=32
LR=5e-5
ATEMP=0.05
EPOCH=2
SEED=21


OUTPUT_DIR=out/mabel-joint-cl-al1-mlm-bs-${BS}-lr-${LR}-msl-${MAX_SEQ}-ep-${EPOCH}
mkdir -p ${OUTPUT_DIR}
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path bert-base-uncased \
    --train_file ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size $BS \
    --learning_rate $LR \
    --max_seq_length $MAX_SEQ \
    --load_best_model_at_end \
    --save_steps 125 \
    --pooler_type cls \
    --align_temp ${ATEMP} \
    --overwrite_output_dir \
    --do_mlm \
    --a1_loss \
    --cl_loss \
    --do_train \
    --seed ${SEED} \
    --fp16 \
    "$@" 2>&1 | tee ${OUTPUT_DIR}/log.txt
