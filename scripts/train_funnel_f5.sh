set -Eeuox
DATE=0426
DECODER=softmax
BS=16
MAX_LEN=512
MODEL="funnel-transformer/xlarge"
for FOLD in 3;do
DESC=norm-$FOLD
python ../src/train.py --fold $FOLD --model $MODEL --decoder $DECODER --warmup_ratio 0.05 \
 --trans_lr 1e-5 --epochs 10 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 32 \
 --desc $DESC \
--input_csv /workspace/train_folds.csv \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE}_${DESC}
done
