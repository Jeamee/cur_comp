set -Eeuox
DATE=0501
DECODER=softmax
BS=16
MAX_LEN=512
MODEL="microsoft/deberta-v3-large"
FOLD=0

DESC=pseudo-$FOLD
python ../src/train.py --fold $FOLD --model $MODEL --decoder $DECODER --warmup_ratio 600 \
 --trans_lr 1e-5 --epochs 10 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 32 --clip_grad_norm 1.0 \
 --gradient_ckpt --pseudo_csv /workspace/pseudo_oof${FOLD}.pkl --val_check_interval 300 \
 --desc $DESC \
--input_csv /workspace/train_folds.csv \
--output /workspace/${MODEL#*/}_pseudo_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE}_${DESC}

