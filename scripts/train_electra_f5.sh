set -Eeuox
DATE=0426
DECODER=softmax
BS=32
MAX_LEN=512
MODEL="google/electra-large-discriminator"


for FOLD in 0 1 2 3 4;do
DESC=norm-$FOLD
python ../src/train.py --fold $FOLD --model $MODEL --decoder $DECODER --warmup_ratio 0.05 \
 --trans_lr 1e-5 --epochs 10 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 32 \
 --gradient_ckpt \
 --desc $DESC \
--input_csv /workspace/train_folds.csv \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE}_${DESC}
done
