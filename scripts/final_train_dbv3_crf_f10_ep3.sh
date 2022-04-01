set -Eeuox
FOLD=3
DATE=0401
DECODER=softmax
BS=32
MAX_LEN=512
MODEL="google/electra-large-discriminator"
DESC="add_special_tokens_bs32"

python ../src/train.py --fold $FOLD --model $MODEL --decoder $DECODER --warmup_ratio 0.05 \
 --trans_lr 1e-5 --epochs 10 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 32 \
 --desc $DESC \
--input_csv /workspace/train_folds.csv \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_oof${FOLD}_${DATE}_${DESC}
