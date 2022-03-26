set -Eeuox
EPOCH=3
DATE=0326
DECODER=softmax
BS=2
MAX_LEN=256
MODEL="microsoft/deberta-v3-large"

python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --warmup_ratio 0.001 \
 --trans_lr 5e-6 --epochs 10 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--input ~/nbme-f7-0326/train_folds.csv\
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE}
