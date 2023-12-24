DATASET=$1;
TRAIN_SAMPLES_RATIO=1.0;
TEST_SAMPLES_RATIO=1.0;

# For both video dataset, we split training and testing dataset
# from the same training dataset online
if [ $1 = "core50" ] ; then
    TRAIN_SAMPLES_RATIO=0.2;
fi

if [ $1 = "stream51" ] ; then
    TRAIN_SAMPLES_RATIO=0.2;
fi


(( python3 main_linear.py \
    --dataset "$DATASET" \
    --size 32 \
    --split_strategy data \
    --train_samples_ratio "$TRAIN_SAMPLES_RATIO" \
    --test_samples_ratio "$TEST_SAMPLES_RATIO" \
    --load_at_beginning \
    --encoder resnet18 \
    --max_epochs 50 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 2048 \
    --num_workers 8 \
    --name "$CKPT_DIR"-"$DATASET" \
    --ckpt_folder "$CKPT_DIR"/"$DATASET"_models \
    --project ever-learn \
    --entity unitn-mhug) 2>&1 ) | \
    tee log_"$CKPT_DIR"_"$DATASET"
