set -Eeuox
DATE=0426
DECODER=softmax
BS=8
MAX_LEN=512
MODEL="microsoft/deberta-v3-large"


for FOLD in 0 1 2 3 4 5;do
DESC=norm-fold${FOLD}
python ../src/train.py --fold $FOLD --model $MODEL --decoder $DECODER --warmup_ratio 0.05 \
 --trans_lr 1e-5 --epochs 10 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 32 \
 --desc $DESC \
--input_csv /workspace/train_folds.csv \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}__${DATE}_${DESC}
done
