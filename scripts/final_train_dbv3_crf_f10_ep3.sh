set -Eeuox
EPOCH=3
DATE=0326
DECODER=softmax
BS=16
MAX_LEN=512
MODEL="microsoft/deberta-v3-large"

python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --warmup_ratio 0.05 \
 --trans_lr 1e-5 --epochs 10 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--input /workspace/train_folds.csv \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE}
